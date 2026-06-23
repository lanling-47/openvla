"""
OpenVLA inference with TensorRT + Temporal Speculative Decoding.

Pipeline:
  Image -> VisionEncoder (TRT or HF) -> Projector -> [visual_embeds + text_embeds]
       -> LLM Backbone (TRT-LLM or HF) with Temporal SpecDec
       -> 7 action tokens -> 7-DOF robot action

Two modes:
  1. Full TRT: vision encoder + LLM both use TensorRT engines
  2. Hybrid TRT: vision encoder in TRT, LLM in HF (fallback)
  3. Full HF: everything in PyTorch (baseline)

Usage:
  # Full TRT (requires both engines built)
  python deploy/trt_inference.py --engine-dir ./trt_engines --model /root/models/openvla-7b

  # Benchmark mode (30 frames)
  python deploy/trt_inference.py --engine-dir ./trt_engines --num-frames 30
"""
import torch
import torch.nn.functional as F
import numpy as np
import time
import os
from PIL import Image
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TRTActionResult:
    """Result of VLA inference."""
    action_tokens: List[int]
    action_values: List[float]
    latency_ms: float
    num_accepted: int
    num_llm_calls: int
    method: str  # "vanilla" or "temporal_spec"
    backend: str  # "trt" or "pytorch"


class TRT_VLA_Inference:
    """
    OpenVLA inference engine with TensorRT acceleration + Temporal SpecDec.

    Automatically detects available engines and falls back gracefully:
      - If TRT vision engine exists -> use it, else use HF vision encoder
      - If TRT-LLM engine exists -> use it, else use HF LLM
      - Temporal SpecDec works with both backends
    """

    def __init__(self, model_path: str, engine_dir: str = None,
                 device: str = "cuda:0"):
        self.device = device
        self.model_path = model_path
        self.prev_action_tokens = None

        # Load HF components (always needed for processor + embedding lookup)
        from transformers import AutoProcessor, AutoModelForVision2Seq

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True)

        self.hf_model = AutoModelForVision2Seq.from_pretrained(
            model_path, torch_dtype=torch.float16,
            low_cpu_mem_usage=True, trust_remote_code=True,
        ).to(device).eval()

        self.vision_backbone = self.hf_model.vision_backbone
        self.projector = self.hf_model.projector
        self.llm_hf = self.hf_model.language_model

        # Detect and load TRT engines
        self.vision_trt = None
        self.llm_trt = None

        if engine_dir is not None:
            self._load_trt_engines(engine_dir)

    def _load_trt_engines(self, engine_dir: str):
        """Load available TensorRT engines."""
        vision_engine_path = os.path.join(engine_dir, "vision.engine")
        llm_engine_dir = os.path.join(engine_dir, "llm")

        if os.path.exists(vision_engine_path):
            try:
                from trt_engine import VisionEngine
                self.vision_trt = VisionEngine(vision_engine_path, self.device)
                print(f"  TRT vision engine loaded: {vision_engine_path}")
            except Exception as e:
                print(f"  Failed to load vision engine: {e}")

        if os.path.exists(os.path.join(llm_engine_dir, "rank0.engine")):
            try:
                from trt_engine import LLMEngine
                self.llm_trt = LLMEngine(llm_engine_dir, device=self.device)
                print(f"  TRT-LLM engine loaded: {llm_engine_dir}")
            except Exception as e:
                print(f"  Failed to load LLM engine: {e}")

    @torch.inference_mode()
    def _encode_vision(self, image_array: np.ndarray,
                       instruction: str) -> torch.Tensor:
        """
        Run vision encoder + projector, return LLM input embeddings.

        Returns: [1, 275, hidden_size] float16 tensor
                 (256 visual tokens + ~19 text tokens)
        """
        img = Image.fromarray(image_array)
        prompt = "In: What action should the robot take to %s?\nOut:" % instruction
        inputs = self.processor(prompt, img).to(self.device, dtype=torch.float16)

        # Vision encode
        if self.vision_trt is not None:
            projected = self.vision_trt.infer(inputs["pixel_values"])
        else:
            patch_features = self.vision_backbone(inputs["pixel_values"])
            projected = self.projector(patch_features)

        # Text embed
        text_embeds = self.llm_hf.get_input_embeddings()(inputs["input_ids"])

        return torch.cat([projected, text_embeds], dim=1)

    def _llm_forward(self, input_embeds: torch.Tensor,
                     input_ids: torch.Tensor = None,
                     past_key_values=None, use_cache: bool = True):
        """
        Single LLM forward pass. Routes to TRT-LLM or HF.

        Args:
            input_embeds: [1, seq_len, hidden_size] (used for prefill)
            input_ids: [1, seq_len] int tensor (used for decode steps)
            past_key_values: KV cache (HF mode only)
            use_cache: whether to return KV cache

        Returns:
            logits, past_key_values
        """
        if self.llm_trt is not None and input_ids is not None:
            # TRT-LLM path (token IDs)
            output = self.llm_trt.decode_step(input_ids, {})
            return output, None

        # HF path (embeddings or token IDs)
        if input_embeds is not None:
            out = self.llm_hf(inputs_embeds=input_embeds, use_cache=use_cache,
                               past_key_values=past_key_values)
        else:
            out = self.llm_hf(input_ids=input_ids, use_cache=use_cache,
                               past_key_values=past_key_values)
        return out.logits, out.past_key_values

    @torch.inference_mode()
    def _vanilla_decode(self, embeds: torch.Tensor,
                        n_tokens: int = 7) -> Tuple[List[int], int]:
        """Standard autoregressive decode (baseline)."""
        logits, kv = self._llm_forward(embeds)
        tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = [tok.item()]
        llm_calls = 1

        for _ in range(n_tokens - 1):
            logits, kv = self._llm_forward(None, tok, kv)
            tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens.append(tok.item())
            llm_calls += 1

        return tokens, llm_calls

    @torch.inference_mode()
    def _temporal_spec_decode(self, embeds: torch.Tensor,
                               prev_tokens: List[int],
                               n_tokens: int = 7) -> Tuple[List[int], int, int]:
        """
        Temporal Speculative Decoding.

        Algorithm:
          1. Prefill current frame -> first token (1 LLM call)
          2. Concatenate first + draft tokens (from prev frame)
          3. Verify all in ONE batched forward pass (1 LLM call)
          4. Accept matching prefix, reject rest
          5. Fill remaining with vanilla decode

        Key insight: robot actions change slowly between frames,
        so prev frame's tokens are a good draft (~52% accept rate).
        """
        # Step 1: Prefill -> first token
        logits, kv = self._llm_forward(embeds)
        first = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = [first.item()]
        llm_calls = 1
        accepted = 0

        if prev_tokens is not None and len(prev_tokens) >= n_tokens:
            remaining = n_tokens - 1
            draft = [torch.tensor([[t]], device=self.device)
                     for t in prev_tokens[1:remaining + 1]]

            # Step 2-3: Batched verify in ONE forward pass
            verify_input = torch.cat([first] + draft, dim=1)
            verify_logits, kv = self._llm_forward(None, verify_input, kv)
            llm_calls += 1

            # Step 4: Accept/reject
            for i in range(remaining):
                target_pred = verify_logits[:, i, :].argmax(dim=-1).item()
                draft_pred = draft[i].item()
                if target_pred == draft_pred:
                    accepted += 1
                    generated.append(target_pred)
                    if len(generated) >= n_tokens:
                        break
                else:
                    generated.append(target_pred)
                    break

            # Bonus token if all draft accepted
            if accepted == remaining and len(generated) < n_tokens:
                bonus = verify_logits[:, remaining, :].argmax(dim=-1).item()
                generated.append(bonus)

            # Step 5: Fill remaining with vanilla decode
            last_tok = torch.tensor([[generated[-1]]], device=self.device)
            while len(generated) < n_tokens:
                logits, kv = self._llm_forward(None, last_tok, kv)
                tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated.append(tok.item())
                last_tok = tok
                llm_calls += 1
        else:
            # No draft available, vanilla decode
            tok = first
            for _ in range(n_tokens - 1):
                logits, kv = self._llm_forward(None, tok, kv)
                tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated.append(tok.item())
                llm_calls += 1

        return generated, llm_calls, accepted

    def predict_action(self, image: np.ndarray, instruction: str,
                       use_spec_decode: bool = True) -> TRTActionResult:
        """
        Predict robot action from image + instruction.

        Args:
            image: RGB image array [H, W, 3] uint8
            instruction: natural language instruction
            use_spec_decode: enable temporal speculative decoding

        Returns:
            TRTActionResult with action tokens, values, and profiling
        """
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Vision encode
        embeds = self._encode_vision(image, instruction)

        # LLM decode
        if use_spec_decode and self.prev_action_tokens is not None:
            tokens, llm_calls, accepted = self._temporal_spec_decode(
                embeds, self.prev_action_tokens, 7)
            method = "temporal_spec"
        else:
            tokens, llm_calls = self._vanilla_decode(embeds, 7)
            accepted = 0
            method = "vanilla"

        torch.cuda.synchronize()
        latency = (time.perf_counter() - t0) * 1000

        # Update temporal cache
        self.prev_action_tokens = tokens

        # Decode action tokens: 256-bin discretization -> normalized [-1, 1]
        action_values = [(t - 31744) / 128.0 - 1.0 for t in tokens]

        backend = "trt" if self.llm_trt is not None else "pytorch"

        return TRTActionResult(
            action_tokens=tokens,
            action_values=action_values,
            latency_ms=latency,
            num_accepted=accepted,
            num_llm_calls=llm_calls,
            method=method,
            backend=backend,
        )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="OpenVLA TensorRT inference")
    parser.add_argument("--model", default="/root/models/openvla-7b")
    parser.add_argument("--engine-dir", default=None,
                        help="TRT engine directory (None = PyTorch only)")
    parser.add_argument("--num-frames", type=int, default=30)
    parser.add_argument("--instruction", default="pick up the red cup")
    parser.add_argument("--no-spec-decode", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("OpenVLA 7B — TensorRT Inference")
    print("GPU: " + torch.cuda.get_device_name(0))
    print("=" * 60)

    engine = TRT_VLA_Inference(args.model, args.engine_dir)

    backend = "TensorRT" if engine.llm_trt is not None else "PyTorch"
    vision = "TensorRT" if engine.vision_trt is not None else "PyTorch"
    print(f"\nBackend: LLM={backend}, Vision={vision}")
    print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Warmup
    np.random.seed(42)
    base_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    def make_frame(base, noise=5):
        n = np.random.randint(-noise, noise + 1, base.shape, dtype=np.int16)
        return np.clip(base.astype(np.int16) + n, 0, 255).astype(np.uint8)

    print("\nWarmup (3 frames)...")
    for _ in range(3):
        engine.predict_action(base_image, args.instruction, use_spec_decode=False)
    torch.cuda.synchronize()

    # Benchmark
    use_spec = not args.no_spec_decode
    print(f"\nRunning {args.num_frames} frames (SpecDec={'ON' if use_spec else 'OFF'})...")

    np.random.seed(42)
    current = base_image.copy()
    times = []
    accepts = []
    calls_list = []

    for f in range(args.num_frames):
        if f > 0:
            current = make_frame(current, 5)

        result = engine.predict_action(current, args.instruction,
                                        use_spec_decode=use_spec)
        times.append(result.latency_ms)
        calls_list.append(result.num_llm_calls)

        if f > 0 and use_spec:
            accepts.append(result.num_accepted)

        if f < 5 or f % 10 == 0:
            acc_str = f"{result.num_accepted}/6" if use_spec else "-"
            print(f"  Frame {f:3d}: {result.latency_ms:7.1f}ms | "
                  f"Calls: {result.num_llm_calls} | Accept: {acc_str} | "
                  f"{result.method}")

    avg_t = sum(times) / len(times)
    avg_a = sum(accepts) / len(accepts) if accepts else 0
    avg_c = sum(calls_list) / len(calls_list)

    print(f"\n{'=' * 60}")
    print(f"Results ({backend} backend)")
    print(f"{'=' * 60}")
    print(f"  Avg latency:    {avg_t:.1f}ms ({1000/avg_t:.1f} Hz)")
    print(f"  Avg LLM calls:  {avg_c:.1f}/frame")
    if use_spec:
        print(f"  Accept rate:    {avg_a:.1f}/6 ({avg_a/6*100:.0f}%)")
    print(f"  SpecDec:        {'ON' if use_spec else 'OFF'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
