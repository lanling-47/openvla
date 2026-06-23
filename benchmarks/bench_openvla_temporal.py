#!/usr/bin/env python3
"""
OpenVLA 7B Speculative Decoding Benchmark — Temporal Draft.

Key insight for VLA: consecutive frames produce similar actions.
Use previous frame's action tokens as draft → no draft model needed!

This is a "free lunch" for robotics:
  - Draft cost = 0 (just reuse cached tokens)
  - Verify cost = 1 target forward pass (instead of 7)
  - Accept rate depends on temporal similarity of actions
"""
import torch, time
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import numpy as np

def main():
    SEP = "=" * 60
    print(SEP)
    print("OpenVLA 7B — Temporal Speculative Decoding")
    print("GPU: " + torch.cuda.get_device_name(0))
    print(SEP)

    processor = AutoProcessor.from_pretrained("/root/models/openvla-7b", trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "/root/models/openvla-7b", torch_dtype=torch.float16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).to("cuda:0").eval()
    llm = model.language_model
    print("OpenVLA loaded: %.2f GB" % (torch.cuda.memory_allocated() / 1e9))

    # Simulate a sequence of robot camera frames
    # In real robotics, consecutive frames are very similar
    np.random.seed(42)
    base_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    def make_frame(base, noise_level=5):
        """Simulate next frame with small perturbation."""
        noise = np.random.randint(-noise_level, noise_level + 1, base.shape, dtype=np.int16)
        return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    prompt = "In: What action should the robot take to pick up the red cup?\nOut:"

    @torch.inference_mode()
    def get_embeds(image_array):
        img = Image.fromarray(image_array)
        inputs = processor(prompt, img).to("cuda:0", dtype=torch.float16)
        pf = model.vision_backbone(inputs["pixel_values"])
        proj = model.projector(pf)
        te = llm.get_input_embeddings()(inputs["input_ids"])
        return torch.cat([proj, te], dim=1)

    @torch.inference_mode()
    def vanilla_decode(embeds, n=7):
        out = llm(inputs_embeds=embeds, use_cache=True)
        kv = out.past_key_values
        tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = [tok.item()]
        for _ in range(n - 1):
            out = llm(tok, past_key_values=kv, use_cache=True)
            kv = out.past_key_values
            tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens.append(tok.item())
        return tokens

    @torch.inference_mode()
    def temporal_spec_decode(embeds, prev_tokens, n=7):
        """
        Speculative decoding using previous frame's action as draft.
        Cost: 1 prefill + 1 verify (vs 1 prefill + 6 decode in vanilla)
        """
        # Prefill
        out = llm(inputs_embeds=embeds, use_cache=True)
        kv = out.past_key_values
        first = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        generated = [first.item()]
        t_calls = 1
        accepted = 0

        if len(generated) >= n:
            return generated, t_calls, accepted, 0

        # Use previous frame's tokens[1:] as draft
        remaining = min(len(prev_tokens) - 1, n - 1)
        draft_toks = [torch.tensor([[t]], device="cuda:0") for t in prev_tokens[1:remaining + 1]]

        if not draft_toks:
            # No draft available, fall back to vanilla
            tok = first
            while len(generated) < n:
                out = llm(tok, past_key_values=kv, use_cache=True)
                kv = out.past_key_values
                tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated.append(tok.item())
                t_calls += 1
            return generated, t_calls, 0, 0

        # Verify in ONE pass (draft is free — no model call needed!)
        verify = torch.cat([first] + draft_toks, dim=1)
        t_out = llm(verify, past_key_values=kv, use_cache=True)
        kv = t_out.past_key_values
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
            last = torch.tensor([[bonus]], device="cuda:0")

        # Fill remaining
        while len(generated) < n:
            t_out = llm(last, past_key_values=kv, use_cache=True)
            kv = t_out.past_key_values
            tok = t_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(tok.item())
            last = tok
            t_calls += 1

        return generated, t_calls, accepted, remaining

    # Warmup
    embeds0 = get_embeds(base_image)
    for _ in range(3):
        vanilla_decode(embeds0, 7)
    torch.cuda.synchronize()

    # === Simulate 20 consecutive frames ===
    print("\nSimulating 20 consecutive robot camera frames...")
    print("(noise_level=5 per pixel, simulating small camera movement)\n")

    num_frames = 20
    vanilla_times = []
    spec_times = []
    accepts = []
    t_calls_list = []
    prev_tokens = None
    current_image = base_image.copy()

    for frame in range(num_frames):
        # Generate next frame (small perturbation)
        if frame > 0:
            current_image = make_frame(current_image, noise_level=5)

        embeds = get_embeds(current_image)

        # Vanilla
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        v_tokens = vanilla_decode(embeds, 7)
        torch.cuda.synchronize()
        v_time = time.perf_counter() - t0
        vanilla_times.append(v_time)

        # Temporal spec decode
        if prev_tokens is not None:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            s_tokens, tc, acc, k = temporal_spec_decode(embeds, prev_tokens, 7)
            torch.cuda.synchronize()
            s_time = time.perf_counter() - t0
            spec_times.append(s_time)
            accepts.append(acc)
            t_calls_list.append(tc)

            # Correctness check
            match = v_tokens == s_tokens
            if frame < 5 or not match:
                print("  Frame %2d: vanilla=%s spec=%s accept=%d/%d match=%s" %
                      (frame, v_tokens, s_tokens, acc, k, match))
        else:
            print("  Frame %2d: vanilla=%s (first frame, no draft)" % (frame, v_tokens))

        prev_tokens = v_tokens

    # Results
    v_avg = sum(vanilla_times) / len(vanilla_times)
    s_avg = sum(spec_times) / len(spec_times) if spec_times else 0
    a_avg = sum(accepts) / len(accepts) if accepts else 0
    tc_avg = sum(t_calls_list) / len(t_calls_list) if t_calls_list else 0
    pct = a_avg / 6 * 100 if accepts else 0

    print("\n" + SEP)
    print("RESULTS: OpenVLA 7B Temporal SpecDec on A100-40GB")
    print(SEP)
    print("  Frames:   %d" % num_frames)
    print("  Vanilla:  %.1f ms = %.1f Hz (7 target calls/frame)" % (v_avg * 1000, 1 / v_avg))
    if s_avg > 0:
        print("  SpecDec:  %.1f ms = %.1f Hz (%.1f target calls/frame)" %
              (s_avg * 1000, 1 / s_avg, tc_avg))
        print("  Speedup:  %.2fx" % (v_avg / s_avg))
        print("  Accept:   %.1f/6 = %.0f%%" % (a_avg, pct))
    print(SEP)

if __name__ == "__main__":
    main()
