import torch
import torch.nn as nn
from typing import Dict, Any, Union


def apply_dynamic_quantization(dia_instance, dtype=torch.qint8):
    """
    Apply dynamic quantization to the Dia model for reduced memory usage and faster CPU inference.

    Args:
        dia_instance: An instance of the Dia class
        dtype: Quantization data type (torch.qint8 or torch.quint8)

    Returns:
        The same instance with a quantized model
    """
    print("Applying dynamic quantization to model...")
    original_model = dia_instance.model

    # Store device for later
    device = next(original_model.parameters()).device

    # Move model to CPU for quantization
    original_model.cpu()

    # List of module types to quantize
    qconfig_dict = {
        # Modules to quantize
        nn.Linear: {},
        nn.Conv1d: {},
        nn.Conv2d: {},
        nn.LSTM: {},
        nn.GRU: {},
    }

    try:
        # Prepare model for quantization
        quantized_model = torch.quantization.quantize_dynamic(
            original_model,
            qconfig_dict,
            dtype=dtype
        )

        # Replace original model with quantized version
        dia_instance.model = quantized_model

        # Move back to original device
        dia_instance.model.to(device)

        # Free up memory
        torch.cuda.empty_cache()

        print(f"Model successfully quantized to {dtype}")
    except Exception as e:
        print(f"Quantization failed: {e}")
        # Move back to original device
        original_model.to(device)

    return dia_instance
