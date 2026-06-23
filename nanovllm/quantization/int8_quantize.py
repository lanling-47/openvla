"""
INT8 Model Quantization — Quantize entire model's Linear layers.

Provides a simple API to quantize any PyTorch model:
  quantize_model(model) -> replaces all nn.Linear with QuantizedLinear

This is the same per-channel absmax INT8 approach used in nano-vllm-project,
applied to VLA models (OpenVLA's Llama-2 7B backbone).
"""

import torch
import torch.nn as nn
from ..kernels.quant_matmul import QuantizedLinear, quantize_weight


def quantize_model(model: nn.Module, skip_patterns=None):
    """
    Quantize all Linear layers in a model to INT8 W8A16.

    Args:
        model: PyTorch model to quantize
        skip_patterns: list of layer name patterns to skip (e.g., ["lm_head", "embed"])

    Returns:
        model: quantized model (in-place modification)
        stats: dict with quantization statistics
    """
    skip_patterns = skip_patterns or ["embed", "lm_head", "norm"]
    count = 0
    total_orig = 0
    total_quant = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        # Skip embedding/head/norm layers
        if any(pat in name for pat in skip_patterns):
            continue

        # Quantize this layer
        q_linear = QuantizedLinear.from_linear(module)
        orig, quant = q_linear.memory_savings()
        total_orig += orig
        total_quant += quant
        count += 1

        # Replace in parent module
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], q_linear)

    stats = {
        "num_quantized_layers": count,
        "original_bytes": total_orig,
        "quantized_bytes": total_quant,
        "compression_ratio": total_orig / max(total_quant, 1),
        "memory_saved_mb": (total_orig - total_quant) / 1e6,
    }
    return model, stats
