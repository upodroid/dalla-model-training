# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Supervised fine-tuning script for decoder language models using Ray Train.

Usage:

# Single node with 8 GPUs
python train.py --config configs/train_recipe/example_recipe.yaml

# Connect to an existing Ray cluster
python train.py --config configs/train_recipe/example_recipe.yaml --ray_address auto
"""

import logging
import os
import sys
from pathlib import Path

import datasets
import ray
import transformers
from ray.train import RunConfig, ScalingConfig
from ray.train.huggingface.transformers import RayTrainReportCallback, prepare_trainer
from ray.train.torch import TorchConfig, TorchTrainer
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from alignment import ScriptArguments, SFTConfig, get_dataset, get_model, get_tokenizer
from trl import ModelConfig, SFTTrainer, TrlParser, get_peft_config, setup_chat_format

from utils import get_model_parameters_summary, format_model_parameters_info

logger = logging.getLogger(__name__)

# Multimodal parameter prefixes that are never used in text-only training.
# Freezing them prevents DDP from failing on un-reduced gradients.
# In Gemma 3 the projector / soft-emb-norm live under `model.*`.
_MULTIMODAL_PREFIXES = (
    "model.vision_tower",
    "model.multi_modal_projector",
    "model.soft_emb_norm",
)


def _freeze_multimodal_params(model):
    """Freeze multimodal branches that are unused during text-only SFT."""
    frozen = 0
    for name, param in model.named_parameters():
        if name.startswith(_MULTIMODAL_PREFIXES):
            param.requires_grad = False
            frozen += param.numel()
    if frozen:
        logger.info("Froze %s multimodal parameters (vision_tower / projector / soft_emb_norm).", f"{frozen:,}")
    return model


def train_func(config):
    script_args = config["script_args"]
    training_args = config["training_args"]
    model_args = config["model_args"]

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load datasets
    ################
    dataset = get_dataset(script_args)
    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args, script_args)
    ############
    # Load model
    ############
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    # Freeze multimodal params so DDP does not choke on unused gradients
    model = _freeze_multimodal_params(model)

    if script_args.resize_token_embeddings:
        tokenizer_vocab_size = len(tokenizer)
        print(f"Resizing model embeddings to {tokenizer_vocab_size}")
        model.resize_token_embeddings(tokenizer_vocab_size)

    if tokenizer.chat_template is None:
        logger.info("No chat template provided, using ChatML.")
        model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")

    unfrozen_layer_patterns = script_args.freeze_except if script_args.freeze_parameters is True else []

    if script_args.freeze_parameters:
        logger.info("*** Freeze layers***")
        for name, param in model.named_parameters():
            if not any(pattern in name for pattern in unfrozen_layer_patterns):
                param.requires_grad = False

    model_parameters_summary = get_model_parameters_summary(model, unfrozen_layer_patterns)
    logger.info(format_model_parameters_info(model_parameters_summary))

    ############################
    # Initialize the SFT Trainer
    ############################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # Add Ray Train reporting callback and prepare for distributed training
    trainer.add_callback(RayTrainReportCallback())
    trainer = prepare_trainer(trainer)

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.model.config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    kwargs = {
        "model_name": training_args.hub_model_id if training_args.push_to_hub else None,
        "dataset_name": script_args.dataset_name,
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    training_args.output_dir = os.path.abspath(training_args.output_dir)

    # Multi-node DDP settings
    if script_args.num_workers > 1:
        if training_args.ddp_find_unused_parameters is None:
            training_args.ddp_find_unused_parameters = True
        if not training_args.ddp_backend:
            training_args.ddp_backend = "nccl"

    repo_root = Path(__file__).resolve().parent
    runtime_env = {
        "py_modules": [
            str(repo_root / "src" / "alignment"),
            str(repo_root / "utils.py"),
        ]
    }

    ray.init(address=script_args.ray_address, runtime_env=runtime_env)

    ray_trainer = TorchTrainer(
        train_func,
        torch_config=TorchConfig(ddp_find_unused_parameters=True),
        train_loop_config={
            "script_args": script_args,
            "training_args": training_args,
            "model_args": model_args,
        },
        scaling_config=ScalingConfig(
            num_workers=script_args.num_workers,
            use_gpu=script_args.num_gpus_per_worker > 0,
            resources_per_worker={"GPU": script_args.num_gpus_per_worker},
        ),
        run_config=RunConfig(
            name="sft-training",
            storage_path=os.path.join(training_args.output_dir, "ray_results"),
        ),
    )
    result = ray_trainer.fit()
    logger.info(f"Training completed. Results: {result}")
