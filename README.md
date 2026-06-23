# VLA-Spec: Vision-Language-Action Model Inference Optimization

Temporal Speculative Decoding + INT8 Quantization + Edge Deployment for OpenVLA 7B, powered by nano-vllm inference engine.

## Overview

VLA-Spec accelerates Vision-Language-Action (VLA) model inference for real-time robot control. Built on **nano-vllm** (a from-scratch LLM inference engine with custom CUDA/Triton kernels), it applies system-level optimizations to OpenVLA 7B — achieving **1.65x end-to-end speedup** on A100 and targeting **20Hz real-time control** on Jetson Orin.

## Architecture

```
                          OpenVLA 7B Inference Pipeline
                          
Image (224x224) --> Vision Encoder (DINOv2+SigLIP, 0.73B) --> 256 visual tokens
                                                                      |
Instruction --> Tokenizer --> 19 text tokens -------------------------+
                                                                      v
                                                             [275 input tokens]
                                                                      |
                                                                      v
                                                   LLM Backbone (Llama-2 7B)
                                                   +-- nano-vllm INT8 Quantization (48% compression)
                                                   +-- nano-vllm PagedAttention (Triton kernel)
                                                   +-- Temporal Speculative Decoding (1.65x speedup)
                                                                      |
                                                                      v
                                                   7 Action Tokens (256-bin x 7-DOF)
                                                                      |
                                                                      v
                                                   Robot Action [dx, dy, dz, rx, ry, rz, gripper]
```

## Key Results (A100-PCIE-40GB, Measured)

| Configuration | Latency | Freq | Speedup | Accept Rate | LLM Calls |
|---|---|---|---|---|---|
| FP16 Vanilla (baseline) | 248 ms | 4.0 Hz | 1.00x | - | 7.0/frame |
| **FP16 + Temporal SpecDec** | **150 ms** | **6.6 Hz** | **1.65x** | 52% | 4.2/frame |
| **nano-vllm INT8 + SpecDec** | **150 ms** | **6.6 Hz** | **1.65x** | 52% | 4.2/frame |
| Stable frames only | 100 ms | 10.0 Hz | 2.50x | 100% | 2.0/frame |

**INT8 quantization**: LLM weight memory 13.5GB -> 7.0GB (48% reduction). Prefill-stage cuBLAS INT8 Tensor Core GEMM 1.66x faster (per-op level). End-to-end speed parity with FP16 on A100; designed for TensorRT INT8 acceleration on edge devices.

### Per-Frame Analysis (30 consecutive frames)

| Phase | Frames | Accept Rate | Speedup | Behavior |
|---|---|---|---|---|
| Smooth motion | 1-4 | 100% (6/6) | 2.5x | All draft tokens accepted |
| Action change | 5-6 | 0% | 1.0x | Automatic fallback to vanilla decode |
| Recovery | 7-9 | 33% (2/6) | 1.3x | Gradually re-converging |
| Steady state | 10+ | ~50% | 1.5x | Mix of accept/reject |

## Optimizations

### 1. Temporal Speculative Decoding (Novel Contribution)

Standard speculative decoding requires a smaller draft model (extra memory + compute). We exploit the **temporal continuity of robot actions**: consecutive camera frames produce nearly identical actions due to physical continuity.

- **Draft source**: previous frame's action tokens (zero compute cost)
- **Verify**: target LLM checks all 6 draft tokens in ONE batched forward pass
- **Fallback**: seamless vanilla decode on action discontinuity
- **vs. Spec-VLA (EMNLP 2025)**: no draft model needed, no training, no extra memory

Key finding: layer-pruned draft models fail for VLA — action tokens concentrate in bins 31744-31999, pruned models output garbage. Temporal draft exploits domain-specific structure instead.

### 2. INT8 W8A16 Weight Quantization (from nano-vllm)

Per-channel absmax INT8 quantization applied to all 224 Linear layers of Llama-2 7B backbone:

- Weight compression: FP16 13.5GB -> INT8 7.0GB (**48% reduction**)
- Prefill: cuBLAS INT8 GEMM via `torch._int_mm` (**1.66x GEMM-level speedup**)
- Decode: cached FP16 weights (no dequant overhead, native cuBLAS speed)
- Purpose: memory reduction for edge deployment + TensorRT INT8 ready

### 3. PagedAttention (Triton kernel, from nano-vllm)

- Paged KV cache management with Triton `store_kvcache_kernel`
- GQA (Grouped Query Attention) support for Llama-2
- SDPA prefill with causal/prefix cache masking
- Designed for multi-robot concurrent inference scenarios

### 4. Edge Deployment Pipeline (Jetson Orin)

Complete ONNX -> TensorRT deployment pipeline:

```bash
# Export on server
python deploy/export_onnx.py --model /path/to/openvla-7b

# Build TensorRT engine on Jetson (INT8 + FP16 fallback)
trtexec --onnx=openvla_llm.onnx --saveEngine=openvla_llm.trt \
  --fp16 --int8 --workspace=4096

# Run inference
python deploy/jetson_inference.py --engine openvla_llm.trt
```

Estimated Jetson AGX Orin performance (INT8 TensorRT + SpecDec): **~48ms = 20.8 Hz**

## Project Structure

```
VLA-Spec/
├── nanovllm/                              # nano-vllm inference engine core
│   ├── kernels/
│   │   ├── paged_attention.py             # PagedAttention (Triton + SDPA)
│   │   └── quant_matmul.py               # INT8 W8A16 fused matmul kernel
│   ├── decoding/
│   │   └── temporal_spec_decode.py       # Temporal Speculative Decoding
│   └── quantization/
│       └── int8_quantize.py              # Model quantization API
├── vla/
│   └── openvla_engine.py                 # OpenVLA 7B inference engine
├── deploy/
│   ├── export_onnx.py                    # ONNX export (LLM + Vision)
│   └── tensorrt_convert.py               # TensorRT config + Orin guide
├── benchmarks/
│   ├── bench_nanovllm_integration.py     # Full nano-vllm integration benchmark
│   ├── bench_openvla_full.py             # Temporal SpecDec detailed analysis
│   ├── bench_openvla_combined.py         # INT8 + SpecDec combined
│   └── bench_openvla_temporal.py         # Temporal draft validation
└── results/                               # A100 experiment logs
    ├── bench_temporal_specdec.txt
    ├── bench_int8_combined.txt
    └── bench_nanovllm_integration.txt
```

## Quick Start

```python
from vla.openvla_engine import OpenVLAEngine

# Load OpenVLA with nano-vllm INT8 quantization
engine = OpenVLAEngine("/path/to/openvla-7b", quantize=True)

# Robot control loop
while running:
    image = camera.capture()  # [224, 224, 3] uint8
    result = engine.predict_action(image, "pick up the red cup")
    robot.execute(result.action_values)  # 7-DOF action
    # Latency: 150ms (6.6Hz) with SpecDec, 100ms (10Hz) during smooth motion
```

## nano-vllm: The Underlying Engine

This project is built on **nano-vllm**, a from-scratch LLM inference engine implementing:

| Component | Implementation | Purpose |
|---|---|---|
| PagedAttention | Triton kernel | KV cache management, no fragmentation |
| INT8 W8A16 matmul | CUDA extension | Quantized weight inference |
| FP8 KV Cache | CUDA extension | Halve KV cache memory |
| Speculative Decoding | Python + KV cache | Draft-verify acceleration framework |

nano-vllm was developed as a learning project to deeply understand LLM inference internals. VLA-Spec extends it to the embodied AI domain with VLA-specific optimizations.

## Requirements

- Python 3.10+
- PyTorch 2.2+ (CUDA 12.x)
- transformers 4.40.x
- triton >= 2.1
- timm 0.9.10

## License

MIT
