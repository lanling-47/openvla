"""
Jetson Orin Deployment — TensorRT conversion and inference pipeline.

Deployment flow:
  1. Export OpenVLA LLM backbone to ONNX (export_onnx.py)
  2. Convert ONNX to TensorRT engine with INT8 calibration (this file)
  3. Run inference on Jetson Orin (jetson_inference.py)

This file provides the TensorRT conversion commands and configuration.
Requires: TensorRT >= 8.6, trtexec CLI tool.
"""

import os
import subprocess
from dataclasses import dataclass


@dataclass
class TRTConfig:
    """TensorRT conversion configuration for Jetson Orin."""
    onnx_path: str = "openvla_llm.onnx"
    engine_path: str = "openvla_llm.trt"
    precision: str = "int8"          # fp16 or int8
    max_batch_size: int = 1
    max_seq_len: int = 512           # max input sequence length
    max_output_tokens: int = 7       # VLA action tokens
    workspace_mb: int = 4096         # TensorRT workspace
    calib_data: str = "calib_data/"  # INT8 calibration dataset
    
    # Jetson Orin specific
    dla_cores: int = 0               # Use DLA cores (0=disabled)
    gpu_fallback: bool = True        # Allow GPU fallback for unsupported ops


def get_trtexec_command(config: TRTConfig) -> str:
    """
    Generate trtexec command for TensorRT engine building.
    
    Run this on the Jetson Orin device (or cross-compile with matching arch).
    Jetson Orin = sm_87 (Ampere).
    """
    cmd_parts = [
        "trtexec",
        "--onnx=%s" % config.onnx_path,
        "--saveEngine=%s" % config.engine_path,
        "--workspace=%d" % config.workspace_mb,
        "--maxBatch=%d" % config.max_batch_size,
    ]
    
    if config.precision == "int8":
        cmd_parts.extend([
            "--int8",
            "--fp16",  # Allow FP16 fallback for non-quantizable ops
            "--calib=%s" % config.calib_data,
        ])
    elif config.precision == "fp16":
        cmd_parts.append("--fp16")
    
    # Input shape specification for VLA LLM backbone
    # input_embeds: [batch, seq_len, hidden_size=4096]
    cmd_parts.extend([
        "--minShapes=input_embeds:1x1x4096",
        "--optShapes=input_embeds:1x275x4096",   # 256 visual + 19 text tokens
        "--maxShapes=input_embeds:1x512x4096",
    ])
    
    if config.dla_cores > 0:
        cmd_parts.extend([
            "--useDLACore=%d" % (config.dla_cores - 1),
            "--allowGPUFallback" if config.gpu_fallback else "",
        ])
    
    return " \\\n  ".join(cmd_parts)


def print_deployment_guide():
    """Print step-by-step Jetson Orin deployment guide."""
    guide = """
================================================================
OpenVLA Jetson Orin Deployment Guide
================================================================

Prerequisites:
  - NVIDIA Jetson AGX Orin (32GB/64GB) or Orin NX
  - JetPack 6.0+ (TensorRT 8.6+, CUDA 12.2+)
  - Python 3.10, PyTorch 2.1+ (Jetson wheel)

Step 1: Export ONNX (on server with full model)
  $ python deploy/export_onnx.py --model /path/to/openvla-7b --output openvla_llm.onnx

Step 2: Transfer to Jetson
  $ scp openvla_llm.onnx jetson:/workspace/
  $ scp openvla_vision.onnx jetson:/workspace/

Step 3: Build TensorRT engine (on Jetson)
  $ trtexec --onnx=openvla_llm.onnx --saveEngine=openvla_llm.trt \\
    --fp16 --int8 --workspace=4096 \\
    --minShapes=input_embeds:1x1x4096 \\
    --optShapes=input_embeds:1x275x4096 \\
    --maxShapes=input_embeds:1x512x4096

Step 4: Run inference
  $ python deploy/jetson_inference.py --engine openvla_llm.trt

Performance targets (Jetson AGX Orin, INT8):
  - Vision encoder: ~15ms (DINOv2+SigLIP INT8)
  - LLM prefill (275 tokens): ~25ms
  - LLM decode (7 tokens): ~35ms (5ms/token)
  - Total: ~75ms = 13.3 Hz (real-time!)
  
  With Temporal SpecDec:
  - LLM decode: ~15ms (1 verify pass instead of 7)
  - Total: ~55ms = 18.2 Hz

Memory budget (Jetson AGX Orin 32GB):
  - Vision encoder INT8: ~0.4 GB
  - LLM backbone INT8: ~3.5 GB
  - KV cache: ~1.0 GB
  - System overhead: ~2.0 GB
  - Total: ~7 GB (fits comfortably in 32GB)
================================================================
"""
    print(guide)


if __name__ == "__main__":
    config = TRTConfig()
    print("TensorRT conversion command:")
    print(get_trtexec_command(config))
    print()
    print_deployment_guide()
