#!/usr/bin/env python3
"""OpenVLA 7B Speculative Decoding Benchmark on A100."""
import torch, time, copy
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import numpy as np

def main():
    SEP = "=" * 60
    print(SEP)
    print("OpenVLA 7B Speculative Decoding Benchmark")
    print("GPU: " + torch.cuda.get_device_name(0))
    print(SEP)

    # Load model
    processor = AutoProcessor.from_pretrained("/root/models/openvla-7b", trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "/root/models/openvla-7b", torch_dtype=torch.float16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).to("cuda:0").eval()
    llm = model.language_model
    print("OpenVLA loaded: %.2f GB" % (torch.cuda.memory_allocated() / 1e9))

    # Create pruned draft (8 layers)
    draft_llm = copy.deepcopy(llm)
    orig = len(draft_llm.model.layers)
    draft_llm.model.layers = draft_llm.model.layers[:8]
    dp = sum(p.numel() for p in draft_llm.parameters()) / 1e9
    print("Draft: %d -> 8 layers, %.2fB params" % (orig, dp))
    print("Total GPU: %.2f GB" % (torch.cuda.memory_allocated() / 1e9))

    # Prepare input
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    prompt = "In: What action should the robot take to pick up the red cup?\nOut:"
    inputs = processor(prompt, dummy_image).to("cuda:0", dtype=torch.float16)

    # Get LLM embeddings (vision encode + project + merge)
    @torch.inference_mode()
    def get_llm_inputs():
        patch_features = model.vision_backbone(inputs["pixel_values"])
        projected = model.projector(patch_features)
        text_embeds = llm.get_input_embeddings()(inputs["input_ids"])
        return torch.cat([projected, text_embeds], dim=1)

    input_embeds = get_llm_inputs()
    print("LLM input: %s" % str(input_embeds.shape))

    @torch.inference_mode()
    def vanilla_decode(target_llm, embeds, n=7):
        out = target_llm(inputs_embeds=embeds, use_cache=True)
        kv = out.past_key_values
        tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = [tok.item()]
        for _ in range(n - 1):
            out = target_llm(tok, past_key_values=kv, use_cache=True)
            kv = out.past_key_values
            tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens.append(tok.item())
        return tokens

    @torch.inference_mode()
    def spec_decode(draft, target, embeds, n=7, k=6):
        # Prefill target
        t_out = target(inputs_embeds=embeds, use_cache=True)
        t_kv = t_out.past_key_values
        first = t_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # Prefill draft + feed first token
        d_out = draft(inputs_embeds=embeds, use_cache=True)
        d_kv = d_out.past_key_values
        d_out = draft(first, past_key_values=d_kv, use_cache=True)
        d_kv = d_out.past_key_values

        generated = [first.item()]
        t_calls = 1
        accepted = 0
        remaining = min(k, n - 1)

        # Draft generates K tokens
        draft_toks = []
        cur = d_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        draft_toks.append(cur)
        for _ in range(remaining - 1):
            d_out = draft(cur, past_key_values=d_kv, use_cache=True)
            d_kv = d_out.past_key_values
            cur = d_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            draft_toks.append(cur)

        # Target verifies in ONE pass
        verify = torch.cat([first] + draft_toks, dim=1)
        t_out = target(verify, past_key_values=t_kv, use_cache=True)
        t_kv = t_out.past_key_values
        t_calls += 1

        # Accept/reject
        last = first
        for i in range(remaining):
            tp = t_out.logits[:, i, :].argmax(dim=-1).item()
            dp = draft_toks[i].item()
            if tp == dp:
                accepted += 1
                generated.append(tp)
                last = draft_toks[i]
                if len(generated) >= n:
                    break
            else:
                generated.append(tp)
                last = torch.tensor([[tp]], device="cuda:0")
                break

        # Bonus if all accepted
        if accepted == remaining and len(generated) < n:
            bonus = t_out.logits[:, remaining, :].argmax(dim=-1).item()
            generated.append(bonus)
            accepted += 1
            last = torch.tensor([[bonus]], device="cuda:0")

        # Fill remaining with vanilla
        while len(generated) < n:
            t_out = target(last, past_key_values=t_kv, use_cache=True)
            t_kv = t_out.past_key_values
            tok = t_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(tok.item())
            last = tok
            t_calls += 1

        return generated, t_calls, accepted, remaining

    # Warmup
    for _ in range(2):
        vanilla_decode(llm, input_embeds, 7)
        spec_decode(draft_llm, llm, input_embeds, 7, 6)
    torch.cuda.synchronize()

    # Vanilla benchmark
    print("\n--- Vanilla (7B LLM, 7 tokens) ---")
    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tokens = vanilla_decode(llm, input_embeds, 7)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    v_avg = sum(times) / len(times)
    print("  Tokens: %s" % tokens)
    print("  Latency: %.1f ms | %.1f Hz" % (v_avg * 1000, 1 / v_avg))

    # Spec decode benchmark
    print("\n--- Speculative Decode (draft=8L, K=6) ---")
    times, results = [], []
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        toks, tc, acc, k = spec_decode(draft_llm, llm, input_embeds, 7, 6)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        results.append((tc, acc, k))
    s_avg = sum(times) / len(times)
    a_avg = sum(r[1] for r in results) / len(results)
    t_avg = sum(r[0] for r in results) / len(results)
    pct = a_avg / results[0][2] * 100 if results[0][2] > 0 else 0
    print("  Tokens: %s" % toks)
    print("  Latency: %.1f ms | %.1f Hz" % (s_avg * 1000, 1 / s_avg))
    print("  Accept: %.1f/%d (%.0f%%)" % (a_avg, results[0][2], pct))
    print("  Target calls: %.1f (vs 7 vanilla)" % t_avg)
    print("  Speedup: %.2fx" % (v_avg / s_avg))

    # Summary
    print("\n" + SEP)
    print("RESULTS: OpenVLA 7B on A100-40GB")
    print(SEP)
    print("  Vanilla:  %.1f ms = %.1f Hz" % (v_avg * 1000, 1 / v_avg))
    print("  SpecDec:  %.1f ms = %.1f Hz" % (s_avg * 1000, 1 / s_avg))
    print("  Speedup:  %.2fx" % (v_avg / s_avg))
    print("  Accept:   %.0f%%" % pct)
    print(SEP)

if __name__ == "__main__":
    main()
