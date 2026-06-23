#!/usr/bin/env python3
"""
Build TensorRT engines for OpenVLA 7B.

Two engines:
  1. Vision encoder: ONNX -> TensorRT (via trtexec)
  2. LLM backbone: HF model -> TensorRT-LLM (via convert_checkpoint + trtllm-build CLI)

Requirements:
  pip install tensorrt-llm
  # Or: https://nvidia.github.io/TensorRT-LLM/installation/linux.html

Usage:
  # Build both engines (FP16)
  python deploy/build_trt_engine.py --model /root/models/openvla-7b --precision fp16

  # Build with INT8 quantization
  python deploy/build_trt_engine.py --model /root/models/openvla-7b --precision int8

  # Build only LLM engine
  python deploy/build_trt_engine.py --model /root/models/openvla-7b --vision-only
"""
import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path


def build_vision_engine(onnx_path: str, engine_path: str, precision: str = "fp16",
                        workspace_mb: int = 4096):
    """Build TensorRT engine for vision encoder from ONNX (via trtexec)."""
    print(f"[Vision] Building TRT engine: {onnx_path} -> {engine_path}")

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={workspace_mb}",
    ]
    if precision in ("fp16", "int8"):
        cmd.append(f"--{precision}")

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  trtexec failed:\n{result.stderr}")
        sys.exit(1)

    size_gb = os.path.getsize(engine_path) / 1e9
    print(f"  Done: {engine_path} ({size_gb:.2f} GB)")


def prepare_llm_checkpoint(model_path: str, checkpoint_dir: str, precision: str = "fp16"):
    """
    Convert HF model to TensorRT-LLM checkpoint format.

    Uses the official convert_checkpoint.py script from TensorRT-LLM examples.
    """
    llm_path = os.path.join(model_path, "language_model")
    if not os.path.exists(llm_path):
        llm_path = model_path

    print(f"[LLM] Converting checkpoint: {llm_path} -> {checkpoint_dir}")

    # Find TensorRT-LLM convert script
    try:
        import tensorrt_llm
        trtllm_dir = Path(tensorrt_llm.__file__).parent.parent
        convert_script = trtllm_dir / "examples" / "llama" / "convert_checkpoint.py"
        if not convert_script.exists():
            convert_script = trtllm_dir / "examples" / "models" / "llama" / "convert_checkpoint.py"
    except Exception:
        convert_script = None

    if convert_script and convert_script.exists():
        cmd = [
            sys.executable, str(convert_script),
            "--model_dir", llm_path,
            "--output_dir", checkpoint_dir,
            "--dtype", f"float16" if precision in ("fp16", "int8") else precision,
        ]
        if precision == "int8":
            cmd.extend(["--use_smooth_quant", "--per_channel", "--per_token"])

        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Convert failed:\n{result.stderr}")
            print("  Trying alternative conversion...")
            _manual_convert(llm_path, checkpoint_dir, precision)
    else:
        print("  TensorRT-LLM convert script not found, using manual conversion")
        _manual_convert(llm_path, checkpoint_dir, precision)


def _manual_convert(model_path: str, checkpoint_dir: str, precision: str):
    """
    Manual checkpoint conversion fallback.

    Copies HF model files and creates TRT-LLM compatible config.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Copy config
    config_src = os.path.join(model_path, "config.json")
    if os.path.exists(config_src):
        with open(config_src) as f:
            config = json.load(f)
        config["auto_map"] = {}
        with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    # Copy safetensors
    for fname in os.listdir(model_path):
        if fname.endswith((".safetensors", ".bin", ".model", ".json")):
            src = os.path.join(model_path, fname)
            dst = os.path.join(checkpoint_dir, fname)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

    print(f"  Manual convert done: {checkpoint_dir}")


def build_llm_engine(checkpoint_dir: str, engine_dir: str, precision: str = "fp16",
                     max_input_len: int = 512, max_output_len: int = 16,
                     max_batch_size: int = 1, workspace_mb: int = 8192):
    """
    Build TensorRT-LLM engine from checkpoint.

    Uses trtllm-build CLI tool.
    """
    os.makedirs(engine_dir, exist_ok=True)
    print(f"[LLM] Building engine: {checkpoint_dir} -> {engine_dir}")

    cmd = [
        "trtllm-build",
        "--checkpoint_dir", checkpoint_dir,
        "--output_dir", engine_dir,
        "--max_batch_size", str(max_batch_size),
        "--max_input_len", str(max_input_len),
        "--max_output_len", str(max_output_len),
        "--gpt_attention_plugin", "float16",
        "--gemm_plugin", "float16",
        "--rmsnorm_plugin", "float16",
        f"--workspace={workspace_mb}",
    ]

    if precision == "int8":
        cmd.extend([
            "--use_smooth_quant",
            "--per_channel",
            "--per_token",
        ])

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  trtllm-build failed:\n{result.stderr}")
        print("\n  Try building manually:")
        print(f"  trtllm-build --checkpoint_dir {checkpoint_dir} \\")
        print(f"    --output_dir {engine_dir} \\")
        print(f"    --max_input_len {max_input_len} --max_output_len {max_output_len}")
        sys.exit(1)

    # Find engine file
    engine_files = list(Path(engine_dir).glob("*.engine"))
    if not engine_files:
        engine_files = list(Path(engine_dir).glob("rank*.engine"))

    if engine_files:
        size_gb = engine_files[0].stat().st_size / 1e9
        print(f"  Done: {engine_files[0]} ({size_gb:.2f} GB)")
    else:
        print(f"  Warning: no .engine file found in {engine_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engines for OpenVLA")
    parser.add_argument("--model", default="/root/models/openvla-7b",
                        help="Path to OpenVLA model")
    parser.add_argument("--output-dir", default="./trt_engines",
                        help="Output directory for engines")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "int8"],
                        help="Engine precision (int8 requires calibration)")
    parser.add_argument("--max-input-len", type=int, default=512)
    parser.add_argument("--max-output-len", type=int, default=16)
    parser.add_argument("--workspace-mb", type=int, default=8192)
    parser.add_argument("--vision-only", action="store_true")
    parser.add_argument("--llm-only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Vision encoder
    if not args.llm_only:
        vision_onnx = os.path.join(args.output_dir, "vision.onnx")
        vision_engine = os.path.join(args.output_dir, "vision.engine")

        if not os.path.exists(vision_onnx):
            print("Step 1: Exporting vision encoder to ONNX...")
            subprocess.run([
                sys.executable, "deploy/export_onnx.py",
                "--model", args.model,
                "--output-vision", vision_onnx,
                "--llm-only",
            ], check=True)

        build_vision_engine(vision_onnx, vision_engine, args.precision, args.workspace_mb)

    # LLM backbone
    if not args.vision_only:
        checkpoint_dir = os.path.join(args.output_dir, "llm_checkpoint")
        engine_dir = os.path.join(args.output_dir, "llm")

        print("\nStep 2: Converting LLM checkpoint...")
        prepare_llm_checkpoint(args.model, checkpoint_dir, args.precision)

        print("\nStep 3: Building LLM engine...")
        build_llm_engine(checkpoint_dir, engine_dir, args.precision,
                         args.max_input_len, args.max_output_len,
                         workspace_mb=args.workspace_mb)

    print(f"\nAll engines built in: {args.output_dir}")
    print(f"Run inference: python deploy/trt_inference.py --engine-dir {args.output_dir}")


if __name__ == "__main__":
    main()
