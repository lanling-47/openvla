#!/usr/bin/env python3
"""
OpenVLA 7B — Manual INT8 Quantization + Temporal SpecDec Benchmark.

Uses manual per-channel INT8 weight quantization (same approach as nano-vllm's test_quant.py).
Quantized weights stored as INT8 + FP16 scale, dequantized on-the-fly during forward pass.
"""
import torch, time
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import numpy as np

def quantize_linear_int8(linear):
    """Quantize a single linear layer: FP16 weights -> INT8 + scale."""
    w = linear.weight.data.float()
    absmax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = absmax / 127.0
    w_int8 = (w / scale).round().clamp(-128, 127).to(torch.int8)
    return w_int8, scale.squeeze(1).half()

class QuantizedLinear(torch.nn.Module):
    """INT8 quantized linear: dequantize on-the-fly during forward."""
    def __init__(self, w_int8, scale, bias=None):
        super().__init__()
        self.register_buffer("w_int8", w_int8)
        self.register_buffer("scale", scale)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x):
        # Dequantize: INT8 -> FP16
        w_fp16 = self.w_int8.half() * self.scale.unsqueeze(1)
        out = torch.nn.functional.linear(x, w_fp16, self.bias)
        return out

def quantize_model_int8(model):
    """Replace all Linear layers with INT8 quantized versions."""
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            w_int8, scale = quantize_linear_int8(module)
            bias = module.bias.data if module.bias is not None else None
            q_linear = QuantizedLinear(w_int8, scale, bias)
            # Replace module
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], q_linear)
            count += 1
    return model, count

def main():
    SEP = "=" * 70
    print(SEP)
    print("OpenVLA 7B — INT8 W8A16 + Temporal SpecDec")
    print("GPU: " + torch.cuda.get_device_name(0))
    print(SEP)

    processor = AutoProcessor.from_pretrained("/root/models/openvla-7b", trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "/root/models/openvla-7b", torch_dtype=torch.float16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).to("cuda:0").eval()
    llm_fp16 = model.language_model
    fp16_mem = torch.cuda.memory_allocated() / 1e9
    print("FP16 model: %.2f GB" % fp16_mem)

    # INT8 quantize the LLM backbone
    import copy
    llm_int8 = copy.deepcopy(llm_fp16)
    llm_int8, num_quantized = quantize_model_int8(llm_int8)
    llm_int8 = llm_int8.to("cuda:0").eval()
    int8_mem = torch.cuda.memory_allocated() / 1e9
    print("INT8 model: %.2f GB (+%.2f GB, %d layers quantized)" %
          (int8_mem, int8_mem - fp16_mem, num_quantized))

    prompt = "In: What action should the robot take to pick up the red cup?\nOut:"
    np.random.seed(42)
    base_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    def make_frame(base, noise=5):
        n = np.random.randint(-noise, noise + 1, base.shape, dtype=np.int16)
        return np.clip(base.astype(np.int16) + n, 0, 255).astype(np.uint8)

    @torch.inference_mode()
    def get_embeds(img_array):
        img = Image.fromarray(img_array)
        inputs = processor(prompt, img).to("cuda:0", dtype=torch.float16)
        pf = model.vision_backbone(inputs["pixel_values"])
        proj = model.projector(pf)
        te = llm_fp16.get_input_embeddings()(inputs["input_ids"])
        return torch.cat([proj, te], dim=1)

    @torch.inference_mode()
    def vanilla_decode(llm, embeds, n=7):
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
    def spec_decode(llm, embeds, prev_tokens, n=7):
        out = llm(inputs_embeds=embeds, use_cache=True)
        kv = out.past_key_values
        first = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = [first.item()]
        tc, acc = 1, 0

        if prev_tokens and len(prev_tokens) >= 7:
            rem = min(6, n - 1)
            draft = [torch.tensor([[t]], device="cuda:0") for t in prev_tokens[1:rem+1]]
            verify = torch.cat([first] + draft, dim=1)
            t_out = llm(verify, past_key_values=kv, use_cache=True)
            kv = t_out.past_key_values
            tc += 1

            last = first
            for i in range(rem):
                tp = t_out.logits[:, i, :].argmax(dim=-1).item()
                dp = draft[i].item()
                if tp == dp:
                    acc += 1
                    generated.append(tp)
                    last = draft[i]
                    if len(generated) >= n: break
                else:
                    generated.append(tp)
                    last = torch.tensor([[tp]], device="cuda:0")
                    break

            if acc == rem and len(generated) < n:
                bonus = t_out.logits[:, rem, :].argmax(dim=-1).item()
                generated.append(bonus)
                last = torch.tensor([[bonus]], device="cuda:0")

            while len(generated) < n:
                t_out = llm(last, past_key_values=kv, use_cache=True)
                kv = t_out.past_key_values
                tok = t_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated.append(tok.item())
                last = tok
                tc += 1
        else:
            tok = first
            for _ in range(n - 1):
                out = llm(tok, past_key_values=kv, use_cache=True)
                kv = out.past_key_values
                tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated.append(tok.item())
                tc += 1

        return generated, tc, acc

    # Warmup
    e = get_embeds(base_image)
    for _ in range(3):
        vanilla_decode(llm_fp16, e)
        vanilla_decode(llm_int8, e)
    torch.cuda.synchronize()

    # Benchmark 4 configs x 30 frames
    N = 30
    configs = [
        ("FP16 Vanilla",   llm_fp16, False),
        ("FP16 + SpecDec", llm_fp16, True),
        ("INT8 Vanilla",   llm_int8, False),
        ("INT8 + SpecDec", llm_int8, True),
    ]

    results = {}
    for name, llm, use_spec in configs:
        np.random.seed(42)
        cur = base_image.copy()
        times, accepts, tcalls = [], [], []
        prev = None

        for f in range(N):
            if f > 0: cur = make_frame(cur, 5)
            emb = get_embeds(cur)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if use_spec:
                toks, tc, acc = spec_decode(llm, emb, prev, 7)
                if f > 0: accepts.append(acc); tcalls.append(tc)
            else:
                toks = vanilla_decode(llm, emb, 7)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            prev = toks

        avg_t = sum(times) / len(times)
        avg_a = sum(accepts) / len(accepts) if accepts else 0
        avg_tc = sum(tcalls) / len(tcalls) if tcalls else 7
        results[name] = {"t": avg_t, "f": 1/avg_t, "a": avg_a, "tc": avg_tc}
        print("  %-18s %.1f ms | %.1f Hz" % (name, avg_t*1000, 1/avg_t))

    # Summary
    base = results["FP16 Vanilla"]["t"]
    print("\n" + SEP)
    print("RESULTS: OpenVLA 7B on A100-40GB")
    print(SEP)
    print("%-22s %10s %8s %9s %8s %8s" % ("Config", "Latency", "Freq", "Speedup", "Accept", "Calls"))
    print("-" * 68)
    for n in ["FP16 Vanilla", "INT8 Vanilla", "FP16 + SpecDec", "INT8 + SpecDec"]:
        r = results[n]
        sp = base / r["t"]
        acc = "%.0f%%" % (r["a"]/6*100) if r["a"] > 0 else "-"
        print("%-22s %8.1f ms %6.1f Hz %8.2fx %8s %6.1f" %
              (n, r["t"]*1000, r["f"], sp, acc, r["tc"]))
    print("-" * 68)
    best = results["INT8 + SpecDec"]
    print("Best: INT8+SpecDec = %.2fx over FP16 baseline" % (base / best["t"]))
    print("Memory: FP16=%.1fGB, +INT8=%.1fGB" % (fp16_mem, int8_mem - fp16_mem))
    print(SEP)

if __name__ == "__main__":
    main()
