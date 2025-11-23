import deepspeed
import torch

def get_layer_precision(layer):
    """
    Determine the precision of a layer by checking its parameters.
    Returns: str indicating precision ('4-bit', '8-bit', '16-bit', '32-bit', or 'unknown')
    """
    if not hasattr(layer, 'weight'):
        return 'no weights'
    
    weight = layer.weight
    if hasattr(weight, 'quant_state'):
        # For bitsandbytes quantized layers
        return f"{weight.quant_state.as_dict()['quant_type']} {weight.quant_state.as_dict()['dtype']}"
    
    dtype = weight.dtype
    if dtype == torch.float32:
        return '32-bit'
    elif dtype == torch.float16 or dtype == torch.bfloat16:
        return '16-bit'
    elif dtype == torch.int8:
        return '8-bit'
    elif dtype == torch.int4 or dtype == torch.uint4:
        return '4-bit'
    else:
        return f'other ({dtype})'

def get_model_parameters_summary(loaded_model, unfrozen_layer_patterns):
    # First pass: Calculate total parameters
    total_params = 0
    for name, param in loaded_model.named_parameters():
        with deepspeed.zero.GatheredParameters(param, modifier_rank=0):
            num_params = param.numel() if param is not None else 0
            total_params += num_params

    # Data structures to collect information
    layer_details = []
    high_level_summaries = []
    trainable_params = 0
    frozen_layers_count = 0
    trainable_layers_count = 0
    layer_param_accumulator = 0
    current_high_level_layer = None

    # Second pass: Process parameters and calculate percentages
    for name, param in loaded_model.named_parameters():
        with deepspeed.zero.GatheredParameters(param, modifier_rank=0):
            num_params = param.numel() if param is not None else 0

            # Layer status (trainable or frozen)
            layer_status = "Trainable" if param.requires_grad else "Frozen"

            # Calculate percentage of total parameters for this layer
            percentage = 100 * num_params / total_params if total_params != 0 else 0

            # Check if we have moved to a new specific high-level layer
            high_level_layer = ".".join(name.split(".")[:3])

            if current_high_level_layer is not None and current_high_level_layer != high_level_layer:
                # Calculate the percentage of total parameters for the previous high-level layer
                high_level_percentage = 100 * layer_param_accumulator / total_params
                high_level_summaries.append(
                    f"Total parameters in {current_high_level_layer}: {layer_param_accumulator} "
                    f"({high_level_percentage:.2f}% of total)"
                )
                layer_param_accumulator = 0  # Reset accumulator for the new layer

            # Update the current high-level layer
            current_high_level_layer = high_level_layer

            # Accumulate parameters for the current high-level layer
            layer_param_accumulator += num_params

            # Collect layer information
            layer_details.append(
                f"Layer: {name} | Status: {layer_status} | Parameters: {num_params} "
                f"({percentage:.2f}% of total)"
            )

            # Count frozen and trainable layers
            if param.requires_grad:
                trainable_layers_count += 1
                trainable_params += num_params
            else:
                frozen_layers_count += 1

    # Add accumulated parameters for the last high-level layer
    if current_high_level_layer is not None:
        high_level_percentage = 100 * layer_param_accumulator / total_params
        high_level_summaries.append(
            f"Total parameters in {current_high_level_layer}: {layer_param_accumulator} "
            f"({high_level_percentage:.2f}% of total)"
        )

    # Collect precision information
    precision_details = []
    for name, module in loaded_model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            precision = get_layer_precision(module)
            precision_details.append(f"Layer: {name} Precision: {precision}")

    # Build summary statistics
    trainable_percentage = 100 * trainable_params / total_params if total_params != 0 else 0
    
    # Return dictionary with all information
    return {
        'layer_details': layer_details,
        'high_level_summaries': high_level_summaries,
        'frozen_layers_count': frozen_layers_count,
        'trainable_layers_count': trainable_layers_count,
        'unfrozen_layer_patterns': unfrozen_layer_patterns,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'trainable_percentage': trainable_percentage,
        'precision_details': precision_details
    }

def format_model_parameters_info(info_dict):
    """
    Format the model parameters dictionary into a string that can be logged.
    
    Args:
        info_dict: Dictionary returned by inspect_model_parameters
        
    Returns:
        Formatted string with all model parameter information
    """
    lines = []
    
    # Add layer details with high-level summaries interspersed
    for i, layer_detail in enumerate(info_dict['layer_details']):
        lines.append(layer_detail)
        # Check if we need to add a high-level summary after this layer
        # This matches the original logic where summaries appear after layer groups
        for summary in info_dict['high_level_summaries']:
            if i < len(info_dict['layer_details']) - 1:
                # Check if the next layer is from a different high-level group
                current_layer = layer_detail.split("|")[0].replace("Layer: ", "").strip()
                next_layer = info_dict['layer_details'][i + 1].split("|")[0].replace("Layer: ", "").strip()
                current_hl = ".".join(current_layer.split(".")[:3])
                next_hl = ".".join(next_layer.split(".")[:3])
                if current_hl in summary and current_hl != next_hl:
                    lines.append(summary)
                    lines.append("")
                    break
    
    # Add the last high-level summary
    if info_dict['high_level_summaries']:
        lines.append(info_dict['high_level_summaries'][-1])
        lines.append("")
    
    # Add summary statistics
    lines.append(f"Number of frozen layers: {info_dict['frozen_layers_count']}")
    lines.append(f"Number of trainable layers: {info_dict['trainable_layers_count']}")
    lines.append("")
    lines.append("Unfrozen layers (patterns):")
    for pattern in info_dict['unfrozen_layer_patterns']:
        lines.append(f"- {pattern}")
    lines.append("")
    lines.append(f"Total parameters: {info_dict['total_params']}")
    lines.append(f"Trainable parameters: {info_dict['trainable_params']}")
    lines.append(f"Percentage of trainable parameters: {info_dict['trainable_percentage']:.2f}%")
    lines.append("")
    
    # Add precision overview
    lines.append("=== Model Precision Overview ===")
    for precision_detail in info_dict['precision_details']:
        lines.append(precision_detail)
    
    return "\n".join(lines)