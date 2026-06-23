"""
INT8 W8A16 Quantized Matrix Multiplication.

Per-channel INT8 weight quantization with FP16 activation:
  - Weights stored as INT8 + per-output-channel FP16 scale
  - Forward: dequantize on-the-fly (INT8 * scale) then matmul with FP16 input
  - Memory reduction: ~50% (INT8 + scale vs FP16)

When CUDA extension (quant_matmul_ext) is available, uses fused kernel
for better performance. Otherwise falls back to PyTorch dequant + matmul.

Ported from nano-vllm-project/test_quant.py.
"""

import torch
import torch.nn as nn

# Try loading custom CUDA kernel
try:
    import quant_matmul_ext
    HAS_FUSED_KERNEL = True
except ImportError:
    HAS_FUSED_KERNEL = False


def quantize_weight(w: torch.Tensor):
    """
    Quantize FP16 weight to INT8 + per-output-channel scale.

    Args:
        w: [N, K] weight matrix (N=output_features, K=input_features)

    Returns:
        w_int8: [N, K] int8 quantized weights
        w_scale: [N] float16 per-channel scale factors
    """
    w_float = w.float()
    absmax = w_float.abs().amax(dim=1, keepdim=True)  # [N, 1]
    scale = (absmax / 127.0).clamp(min=1e-8)  # [N, 1]
    w_int8 = (w_float / scale).round().clamp(-128, 127).to(torch.int8)
    w_scale = scale.squeeze(1).to(torch.float16)  # [N]
    return w_int8, w_scale


class QuantizedLinear(nn.Module):
    """
    INT8 quantized linear layer (W8A16).

    Stores weights as INT8 + FP16 scale. Forward pass dequantizes
    on-the-fly and computes matmul with FP16 input.

    If fused CUDA kernel available: uses quant_matmul_ext.fused_w8a16_matmul
    Otherwise: dequantize to FP16 then torch.nn.functional.linear
    """

    def __init__(self, w_int8, w_scale, bias=None):
        super().__init__()
        self.register_buffer("w_int8", w_int8)      # [N, K] int8
        self.register_buffer("w_scale", w_scale)    # [N] float16
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None
        self.out_features = w_int8.shape[0]
        self.in_features = w_int8.shape[1]

    def forward(self, x):
        if HAS_FUSED_KERNEL and x.is_cuda:
            # Fused kernel: INT8 dequant + matmul in one pass
            batch_shape = x.shape[:-1]
            x_2d = x.reshape(-1, self.in_features)
            output = torch.empty(x_2d.shape[0], self.out_features,
                                 dtype=torch.float16, device=x.device)
            quant_matmul_ext.fused_w8a16_matmul(output, x_2d, self.w_int8, self.w_scale)
            if self.bias is not None:
                output = output + self.bias
            return output.reshape(*batch_shape, self.out_features)
        else:
            # PyTorch fallback: dequant then matmul
            w_fp16 = self.w_int8.half() * self.w_scale.unsqueeze(1)
            return nn.functional.linear(x, w_fp16, self.bias)

    @staticmethod
    def from_linear(linear: nn.Linear) -> "QuantizedLinear":
        """Convert a standard Linear layer to quantized version."""
        w_int8, w_scale = quantize_weight(linear.weight.data)
        bias = linear.bias.data if linear.bias is not None else None
        return QuantizedLinear(w_int8, w_scale, bias)

    def memory_savings(self):
        """Return (original_bytes, quantized_bytes) for this layer."""
        orig = self.out_features * self.in_features * 2  # FP16
        quant = self.out_features * self.in_features * 1 + self.out_features * 2  # INT8 + scale
        return orig, quant
