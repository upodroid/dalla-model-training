
# Dalla training recipe

> Fork of [alignment-handbook](https://github.com/huggingface/alignment-handbook) 



1. Follow the [installation instructions](#installation-instructions) to set up your environment etc.

## Contents

## Dalla pipeline:

### 1- [dalla-data-processing](https://github.com/U4RASD/dalla-data-processing)
A comprehensive Arabic data processing pipeline with deduplication, stemming, quality checking, readability scoring abd dataset packing, used for the DALLA Models.

### 2- [dalla-sentencepiece-tokenizer-manipulation](https://github.com/U4RASD/dalla-sentencepiece-tokenizer-manipulation)
Extend existing tokenizers with new vocabulary from custom training data. Train a new tokenizer on your domain-specific data and merge it with any sentencepiece tokenizer (e.g., Gemma) by replacing unused language tokens.

### 3- [r-bpe](https://github.com/U4RASD/r-bpe)
Extend existing tokenizers with new vocabulary from custom training data. Train a new tokenizer on your domain-specific data and merge it with any bpe tokenizer (e.g., Llama) by replacing unused language tokens. and is also needed to use the r-bpe tokenizers

## How to use Dalla pipeline
1- Start by using [dalla-data-processing](https://github.com/U4RASD/dalla-data-processing) for data cleaning, ([deduplication](https://github.com/U4RASD/dalla-data-processing/blob/main/dalla_data_processing/deduplication/README.md), [Quality checking](https://github.com/U4RASD/dalla-data-processing/blob/main/dalla_data_processing/quality/README.md), [Readability scoring](https://github.com/U4RASD/dalla-data-processing/blob/main/dalla_data_processing/readability/README.md), and [stemming](https://github.com/U4RASD/dalla-data-processing/blob/main/dalla_data_processing/stemming/README.md))

2- Edit the original tokenizer of the model you are starting from, if your model is a sentencepiece tokenizer use [dalla-sentencepiece-tokenizer-manipulation](https://github.com/U4RASD/dalla-sentencepiece-tokenizer-manipulation), and if it is a BPE tokenizer use [r-bpe](https://github.com/U4RASD/r-bpe)

3- pack your training data using [dataset-packing](https://github.com/U4RASD/dalla-data-processing/blob/main/dalla_data_processing/packing/README.md) this repo packs the training dataset to fit the maximum sequence length you choose 


## Installation instructions

To run the code in this project, first, create a Python virtual environment using e.g. `uv`:

```shell
uv venv handbook --python 3.11 && source handbook/bin/activate && uv pip install --upgrade pip
```

> [!TIP]
> To install `uv`, follow the [UV Installation Guide](https://docs.astral.sh/uv/getting-started/installation/).

Next, install PyTorch `v2.4.1` 

```shell
uv pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```


You can then install the remaining package dependencies as follows:

```shell
uv pip install .
```

You will also need Flash Attention 2 installed, which can be done by running:

```shell
uv pip install "flash-attn==2.7.4.post1" --no-build-isolation
```

Next, log into your Hugging Face account as follows:

```shell
huggingface-cli login
```

Finally, install Git LFS so that you can push models to the Hugging Face Hub:

```shell
sudo apt-get update; sudo apt-get install git-lfs
```

## Usage

### Training Models

To train a model using CPT (Continued Pre-Training) or SFT (Supervised Fine-Tuning), use the following command:

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/accelerate/zero3.yaml train.py --config configs/train_recipe/example_recipe.yaml
```

### Configuration

#### Accelerate Configuration

Edit `configs/accelerate/zero3.yaml` to configure your hardware setup. For example, to set the number of GPUs:

You need a machine with 8 H100 GPUs to run the example
```yaml
num_processes: 8  # Set to the number of GPUs you have
```

#### Training Recipe Configuration

Edit `configs/train_recipe/example_recipe.yaml` to configure your training parameters:

**Models and Datasets:**
- Use local models or models from Hugging Face Hub
- Use local datasets or datasets from Hugging Face Hub

**Layer Freezing (Optional):**

By default, all model parameters are trained. If you want to freeze the entire model and train only specific layers, add:

```yaml
freeze_parameters: true
freeze_except: 
  - embed_tokens
  - layers.0
```

If `freeze_parameters` is not specified, the entire model will be trained.

**Custom Tokenizer (Optional):**

By default, the model's original tokenizer is used. To use a different tokenizer:

```yaml
tokenizer_name_or_path: "your-tokenizer-name"
```

**R-BPE Tokenizer (Optional):**

If your custom tokenizer is an R-BPE tokenizer:
```yaml
is_rbpe_tokenizer: true
```

If the custom tokenizer has a different vocabulary size than the original model, enable token embedding resizing:

```yaml
resize_token_embeddings: true
```

## Project structure

```
├── configs/
│   ├── accelerate/
│   │   └── zero3.yaml          <- Accelerate configuration (GPU setup, distributed training)
│   └── train_recipe/
│       └── example_recipe.yaml <- Training recipe configuration (model, dataset, hyperparameters)
├── src/
│   └── alignment/              <- Source code for training
│       ├── configs.py
│       ├── data.py
│       ├── model_utils.py
│       └── release.py
├── tests/                      <- Unit tests
├── train.py                    <- Main training script for CPT or SFT
├── setup.py                    <- Makes project pip installable
├── setup.cfg                   <- Installation config
└── README.md                   <- The top-level README for developers using this project
```


