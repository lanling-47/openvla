#!/usr/bin/env python3
"""
VLA-Spec Benchmark: Vanilla vs Speculative Decoding for VLA Action Generation.

Compares:
  1. Vanilla autoregressive decoding (baseline)
  2. Speculative decoding with pruned draft model, different K values

VLA Context:
  - Each "action" = 7 tokens (7-DOF robot arm)
  - Target model: Qwen3-1.7B (simulates VLA backbone)
  - Draft model: Qwen3-0.6B pruned to 4 layers (5-10x faster than target)
"""

import sys
import os
import torch
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vla_spec.models.vla_loader import VLAConfig, VLAModelLoader
from vla_spec.decoding.spec_decode import SpeculativeDecoder, create_pruned_draft_model
from vla_spec.decoding.vanilla_decode import VanillaDecoder


def run_benchmark(args):
    print("=" * 70)
    print("VLA-Spec Benchmark: Speculative Decoding for VLA Action Generation")
    print("=" * 70)

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Compute Capability: sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}")
    print(f"PyTorch: {torch.__version__}")

    # Load target model
    print(f"\nLoading target model: {args.target_model}")
    target_model = VLAModelLoader._load_single(args.target_model, torch.float16, "cuda")
    target_params = sum(p.numel() for p in target_model.parameters()) / 1e9
    print(f"  Target: {target_params:.2f}B params")

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.draft_model, trust_remote_code=True)

    # Create pruned draft model
    draft_model = create_pruned_draft_model(
        args.draft_model, num_layers=args.draft_layers,
        dtype=torch.float16, device="cuda"
    )

    print(f"\nGPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")

    # Prepare input
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Output 7 integers between 0 and 255 for robot arm joint angles:"}],
        tokenize=False, add_generation_prompt=True,
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    print(f"Input length: {input_ids.shape[1]} tokens")
    print(f"Action tokens to generate: {args.action_tokens}")

    num_runs = args.runs
    results = []

    # === Warmup ===
    print("\n--- Warmup ---")
    for _ in range(3):
        with torch.no_grad():
            _ = target_model(input_ids, use_cache=True)
        with torch.no_grad():
            _ = draft_model(input_ids, use_cache=True)
    torch.cuda.synchronize()

    # === 1. Vanilla Decoding ===
    print(f"\n--- Vanilla Decoding ({num_runs} runs) ---")
    vanilla_decoder = VanillaDecoder(target_model, tokenizer)
    vanilla_times = []

    for run in range(num_runs):
        torch.cuda.synchronize()
        result = vanilla_decoder.decode(input_ids, max_new_tokens=args.action_tokens)
        vanilla_times.append(result.total_time)
        if run == 0:
            print(f"  Run 0: {result.total_time*1000:.2f}ms | "
                  f"{result.tokens_per_sec:.1f} tok/s | "
                  f"{result.num_forward_passes} passes")

    avg_vanilla = sum(vanilla_times) / len(vanilla_times)
    std_vanilla = (sum((t-avg_vanilla)**2 for t in vanilla_times) / len(vanilla_times))**0.5
    print(f"  Avg: {avg_vanilla*1000:.2f}ms ± {std_vanilla*1000:.2f}ms | "
          f"{args.action_tokens/avg_vanilla:.1f} tok/s | "
          f"{1/avg_vanilla:.1f} Hz")

    results.append(("vanilla", 0, avg_vanilla, 0, 0, args.action_tokens, 0,
                     args.action_tokens, 0))

    # === 2. Speculative Decoding ===
    spec_decoder = SpeculativeDecoder(draft_model, target_model, tokenizer)

    for k in args.k_values:
        print(f"\n--- Speculative Decoding K={k} ({num_runs} runs) ---")
        spec_times = []
        spec_results = []

        for run in range(num_runs):
            torch.cuda.synchronize()
            result = spec_decoder.decode(input_ids, max_new_tokens=args.action_tokens, k=k)
            spec_times.append(result.total_time)
            spec_results.append(result)
            if run == 0:
                print(f"  Run 0: {result.total_time*1000:.2f}ms | "
                      f"{result.tokens_per_sec:.1f} tok/s | "
                      f"accept: {result.accept_rate:.1f}% | "
                      f"target calls: {result.num_target_calls} | "
                      f"draft calls: {result.num_draft_calls}")

        avg_spec = sum(spec_times) / len(spec_times)
        std_spec = (sum((t-avg_spec)**2 for t in spec_times) / len(spec_times))**0.5
        avg_accept = sum(r.accept_rate for r in spec_results) / len(spec_results)
        avg_target = sum(r.num_target_calls for r in spec_results) / len(spec_results)
        avg_draft = sum(r.num_draft_calls for r in spec_results) / len(spec_results)
        avg_draft_t = sum(r.draft_time for r in spec_results) / len(spec_results)
        avg_verify_t = sum(r.verify_time for r in spec_results) / len(spec_results)
        speedup = avg_vanilla / avg_spec

        print(f"  Avg: {avg_spec*1000:.2f}ms ± {std_spec*1000:.2f}ms | "
              f"accept: {avg_accept:.1f}% | "
              f"speedup: {speedup:.2f}x | "
              f"{1/avg_spec:.1f} Hz")

        results.append(("spec_decode", k, avg_spec, avg_draft_t, avg_verify_t,
                         args.action_tokens, avg_accept, int(avg_target), int(avg_draft)))

    # === Summary ===
    print("\n" + "=" * 110)
    print(f"{'Method':>14} {'K':>4} {'Time(ms)':>10} {'tok/s':>8} {'Accept%':>9} "
          f"{'Target':>7} {'Draft':>7} {'Draft(ms)':>10} {'Verify(ms)':>11} "
          f"{'Speedup':>9} {'Freq(Hz)':>9}")
    print("-" * 110)

    vanilla_time = results[0][2]
    for method, k, t, draft_t, verify_t, tokens, accept, tcalls, dcalls in results:
        speedup = vanilla_time / t if t > 0 else 0
        freq = 1 / t if t > 0 else 0
        tps = tokens / t if t > 0 else 0
        name = "Vanilla" if method == "vanilla" else f"Spec"
        print(f"{name:>14} {k:>4} {t*1000:>10.2f} {tps:>8.1f} {accept:>8.1f}% "
              f"{tcalls:>7} {dcalls:>7} {draft_t*1000:>10.2f} {verify_t*1000:>11.2f} "
              f"{speedup:>8.2f}x {freq:>8.1f}Hz")

    print("=" * 110)

    # === VLA Analysis ===
    print(f"\nVLA Control Frequency (target: ≥10Hz for real-time):")
    for method, k, t, _, _, _, _, _, _ in results:
        freq = 1 / t if t > 0 else 0
        latency = t * 1000
        name = "Vanilla" if method == "vanilla" else f"Spec(K={k})"
        status = "✅" if freq >= 10 else ("⚠️" if freq >= 5 else "❌")
        print(f"  {name:>14}: {latency:>7.1f}ms = {freq:>5.1f}Hz {status}")

    print(f"\nDraft model: Qwen3-0.6B pruned to {args.draft_layers} layers")
    print(f"Target model: Qwen3-1.7B (full)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLA-Spec Benchmark")
    parser.add_argument("--target-model", default="/root/models/Qwen3-1.7B")
    parser.add_argument("--draft-model", default="/root/models/Qwen3-0.6B")
    parser.add_argument("--draft-layers", type=int, default=4,
                        help="Number of layers to keep in pruned draft model")
    parser.add_argument("--action-tokens", type=int, default=7)
    parser.add_argument("--k-values", type=int, nargs="+", default=[3, 4, 7])
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    run_benchmark(args)
