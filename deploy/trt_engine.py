"""
TensorRT engine wrappers for OpenVLA inference.

Two engine types:
  - VisionEngine: plain TensorRT for vision encoder (DINOv2 + SigLIP + Projector)
  - LLMEngine: TensorRT-LLM ModelRunner for LLM backbone (Llama-2 7B)

The LLM engine handles KV cache internally via PagedAttention plugin.
Temporal SpecDec is orchestrated at the Python level using the engine's
batched decode capability.
"""
import torch
import numpy as np
import json
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass

try:
    import tensorrt as trt
    HAS_TRT = True
except ImportError:
    HAS_TRT = False

try:
    import tensorrt_llm
    from tensorrt_llm.runtime import ModelRunner, ModelConfig, SamplingConfig
    HAS_TRTLLM = True
except ImportError:
    HAS_TRTLLM = False


# ============================================================
# Vision Encoder: plain TensorRT
# ============================================================

class VisionEngine:
    """
    TensorRT engine wrapper for vision encoder.

    Input:  pixel_values [1, 6, 224, 224] float16  (DINOv2 3ch + SigLIP 3ch)
    Output: visual_embeddings [1, 256, 4096] float16
    """

    def __init__(self, engine_path: str, device: str = "cuda:0"):
        if not HAS_TRT:
            raise RuntimeError("pip install tensorrt")

        self.device = device
        self.stream = torch.cuda.current_stream().cuda_stream

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self._alloc_buffers()

    def _alloc_buffers(self):
        """Pre-allocate GPU buffers for all I/O tensors."""
        self.buffers = {}
        self.io_names = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype_trt = self.engine.get_tensor_dtype(name)
            dtype_np = trt.nptype(dtype_trt)
            mode = self.engine.get_tensor_mode(name)

            numel = 1
            for d in shape:
                if d > 0:
                    numel *= d

            buf = torch.empty(numel, dtype=torch.from_numpy(
                np.empty(0, dtype=dtype_np)).dtype, device=self.device)

            self.buffers[name] = {
                "tensor": buf,
                "shape": shape,
                "dtype_np": dtype_np,
                "is_input": mode == trt.TensorIOMode.INPUT,
            }
            self.io_names.append(name)

    def infer(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Run vision encoder.

        Args:
            pixel_values: [1, C, H, W] float16 tensor on GPU

        Returns:
            visual_embeddings: [1, 256, hidden_size] float16 tensor on GPU
        """
        import pycuda.driver as cuda

        # Set input
        for name, info in self.buffers.items():
            if info["is_input"]:
                input_data = pixel_values.contiguous()
                cuda.memcpy_dtod(
                    int(info["tensor"].data_ptr()),
                    int(input_data.data_ptr()),
                    input_data.nelement() * input_data.element_size()
                )
                self.context.set_tensor_address(name, int(info["tensor"].data_ptr()))

        # Allocate output
        output_info = None
        for name, info in self.buffers.items():
            if not info["is_input"]:
                output_info = info
                shape = info["shape"]
                # Handle dynamic shapes
                actual_shape = tuple(
                    self.context.get_tensor_shape(name) if d == -1 else d
                    for d in shape
                )
                numel = 1
                for d in actual_shape:
                    numel *= d
                output_buf = torch.empty(numel, dtype=torch.float16, device=self.device)
                info["tensor"] = output_buf
                self.context.set_tensor_address(name, int(output_buf.data_ptr()))

        # Execute
        self.context.execute_async_v3(self.stream)
        torch.cuda.synchronize()

        # Reshape output
        shape = output_info["shape"]
        return output_info["tensor"].reshape(shape)


# ============================================================
# LLM Backbone: TensorRT-LLM
# ============================================================

class LLMEngine:
    """
    TensorRT-LLM engine for Llama-2 7B backbone.

    Uses ModelRunner which manages KV cache internally (PagedAttention).

    Three forward modes:
      1. prefill(input_ids, input_embeds)  -> first token
      2. decode_step(token_ids)            -> next token (vanilla)
      3. decode_verify(token_ids)          -> logits for N tokens (SpecDec verify)

    Note on inputs_embeds:
      TensorRT-LLM natively takes token IDs. For VLA, the LLM receives
      visual embeddings (not token IDs). We handle this by:
        - Building the engine with --use_custom_all_reduce and embedding disabled
        - Passing input_embedding directly to ModelRunner.generate()
        - The engine skips the embedding lookup and uses provided embeddings
    """

    def __init__(self, engine_dir: str, vocab_size: int = 32000,
                 max_batch_size: int = 1, max_input_len: int = 512,
                 max_output_len: int = 16, device: str = "cuda:0"):
        if not HAS_TRTLLM:
            raise RuntimeError("pip install tensorrt-llm")

        self.device = device
        self.vocab_size = vocab_size
        self.engine_dir = engine_dir

        # Load engine config
        config_path = os.path.join(engine_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                self.config = json.load(f)
            self.vocab_size = self.config.get("vocab_size", vocab_size)

        # Create ModelRunner
        self.runner = ModelRunner.from_engine(
            engine_dir=engine_dir,
            rank=0,
        )

    def prefill(self, input_ids: torch.Tensor,
                input_embeds: Optional[torch.Tensor] = None) -> Tuple[int, torch.Tensor]:
        """
        Prefill phase: process full input sequence, return first token.

        Args:
            input_ids: [1, seq_len] int32 tensor (dummy IDs if using embeddings)
            input_embeds: [1, seq_len, hidden_size] float16 (optional, for VLA)

        Returns:
            first_token: int, first generated token ID
            kv_state: opaque KV cache state for subsequent decode calls
        """
        sampling_config = SamplingConfig(
            end_id=-1,
            pad_id=-1,
            max_new_tokens=1,
            temperature=0.0,
            top_k=1,
            top_p=0.0,
        )

        kwargs = {"batch_input_ids": input_ids}
        if input_embeds is not None:
            kwargs["input_embedding"] = input_embeds

        output = self.runner.generate(
            **kwargs,
            sampling_config=sampling_config,
        )
        torch.cuda.synchronize()

        first_token = output["output_ids"][0, 0, -1].item()
        # ModelRunner manages KV cache internally across generate calls
        return first_token, output

    def decode_step(self, token_ids: torch.Tensor,
                    prev_output: dict) -> Tuple[int, dict]:
        """
        Single-token decode step (vanilla fallback).

        Args:
            token_ids: [1, 1] int32, previous token
            prev_output: dict from previous prefill/decode call

        Returns:
            next_token: int
            output: updated output dict
        """
        sampling_config = SamplingConfig(
            end_id=-1,
            pad_id=-1,
            max_new_tokens=1,
            temperature=0.0,
            top_k=1,
            top_p=0.0,
        )

        output = self.runner.generate(
            batch_input_ids=token_ids,
            sampling_config=sampling_config,
        )
        torch.cuda.synchronize()

        next_token = output["output_ids"][0, 0, -1].item()
        return next_token, output

    def decode_verify(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Verify N draft tokens in one forward pass.

        Args:
            token_ids: [1, N] int32, draft tokens to verify

        Returns:
            output_ids: [1, 1, N+1] int32, generated token IDs
        """
        n = token_ids.shape[1]
        sampling_config = SamplingConfig(
            end_id=-1,
            pad_id=-1,
            max_new_tokens=n,
            temperature=0.0,
            top_k=1,
            top_p=0.0,
        )

        output = self.runner.generate(
            batch_input_ids=token_ids,
            sampling_config=sampling_config,
        )
        torch.cuda.synchronize()
        return output
