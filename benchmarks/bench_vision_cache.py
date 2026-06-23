import torch, time, numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

print('=' * 70)
print('OpenVLA 7B — Vision Cache + SpecDec Benchmark')
print('GPU:', torch.cuda.get_device_name(0))
print('=' * 70)

processor = AutoProcessor.from_pretrained('/root/models/openvla-7b', trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    '/root/models/openvla-7b', torch_dtype=torch.float16,
    low_cpu_mem_usage=True, trust_remote_code=True,
).to('cuda:0').eval()
llm = model.language_model
print('Loaded: %.2f GB' % (torch.cuda.memory_allocated()/1e9))

prompt = 'In: What action should the robot take to pick up the red cup?\nOut:'
np.random.seed(42)
base_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

def make_frame(b, noise=5):
    return np.clip(b.astype(np.int16) + np.random.randint(-noise,noise+1,b.shape,dtype=np.int16), 0, 255).astype(np.uint8)

@torch.inference_mode()
def full_encode(img):
    inputs = processor(prompt, Image.fromarray(img)).to('cuda:0', dtype=torch.float16)
    pf = model.vision_backbone(inputs['pixel_values'])
    proj = model.projector(pf)
    te = llm.get_input_embeddings()(inputs['input_ids'])
    return torch.cat([proj, te], dim=1)

@torch.inference_mode()
def spec_decode(target, emb, prev, n=7):
    out = target(inputs_embeds=emb, use_cache=True)
    kv = out.past_key_values
    first = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    gen = [first.item()]
    tc, acc = 1, 0
    if prev and len(prev) >= n:
        rem = min(6, n-1)
        draft = [torch.tensor([[t]], device='cuda:0') for t in prev[1:rem+1]]
        verify = torch.cat([first]+draft, dim=1)
        t_out = target(verify, past_key_values=kv, use_cache=True)
        kv = t_out.past_key_values; tc += 1
        last = first
        for i in range(rem):
            tp = t_out.logits[:, i, :].argmax(dim=-1).item()
            if tp == draft[i].item():
                acc += 1; gen.append(tp); last = draft[i]
                if len(gen) >= n: break
            else:
                gen.append(tp); last = torch.tensor([[tp]], device='cuda:0'); break
        if acc == rem and len(gen) < n:
            bonus = t_out.logits[:, rem, :].argmax(dim=-1).item()
            gen.append(bonus); last = torch.tensor([[bonus]], device='cuda:0')
        while len(gen) < n:
            t_out = target(last, past_key_values=kv, use_cache=True)
            kv = t_out.past_key_values
            tok = t_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen.append(tok.item()); last = tok; tc += 1
    else:
        tok = first
        for _ in range(n-1):
            out = target(tok, past_key_values=kv, use_cache=True)
            kv = out.past_key_values
            tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen.append(tok.item()); tc += 1
    return gen, tc, acc

# Warmup
e = full_encode(base_img)
for _ in range(3): spec_decode(llm, e, None)
torch.cuda.synchronize()

N = 30
print('\nBenchmarking %d frames...\n' % N)
results = {}

# 1. Vanilla
np.random.seed(42); cur = base_img.copy(); times = []
for f in range(N):
    if f > 0: cur = make_frame(cur, 5)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    emb = full_encode(cur)
    out = llm(inputs_embeds=emb, use_cache=True); kv = out.past_key_values
    tok = out.logits[:,-1,:].argmax(dim=-1,keepdim=True)
    for _ in range(6):
        out = llm(tok, past_key_values=kv, use_cache=True); kv = out.past_key_values
        tok = out.logits[:,-1,:].argmax(dim=-1,keepdim=True)
    torch.cuda.synchronize(); times.append(time.perf_counter()-t0)
results['Vanilla'] = sum(times)/len(times)
print('  Vanilla:                %.1f ms | %.1f Hz' % (results['Vanilla']*1000, 1/results['Vanilla']))

# 2. SpecDec only
np.random.seed(42); cur = base_img.copy(); times = []; accs = []; prev = None
for f in range(N):
    if f > 0: cur = make_frame(cur, 5)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    emb = full_encode(cur)
    toks, tc, acc = spec_decode(llm, emb, prev)
    torch.cuda.synchronize(); times.append(time.perf_counter()-t0)
    if f > 0: accs.append(acc)
    prev = toks
results['SpecDec'] = sum(times)/len(times)
a1 = sum(accs)/len(accs) if accs else 0
print('  SpecDec:                %.1f ms | %.1f Hz | accept=%.0f%%' % (results['SpecDec']*1000, 1/results['SpecDec'], a1/6*100))

# 3. VisionCache + SpecDec
np.random.seed(42); cur = base_img.copy(); times = []; accs = []
prev = None; cached_emb = None; prev_frame = None; vis_skipped = 0
for f in range(N):
    if f > 0: cur = make_frame(cur, 5)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    
    # Vision cache: skip if pixel diff small
    if prev_frame is not None:
        diff = np.abs(cur.astype(np.int16) - prev_frame.astype(np.int16)).mean()
        if diff < 5.0:
            emb = cached_emb  # Reuse!
            vis_skipped += 1
        else:
            emb = full_encode(cur)
            cached_emb = emb
    else:
        emb = full_encode(cur)
        cached_emb = emb
    prev_frame = cur.copy()
    
    toks, tc, acc = spec_decode(llm, emb, prev)
    torch.cuda.synchronize(); times.append(time.perf_counter()-t0)
    if f > 0: accs.append(acc)
    prev = toks
results['VisionCache+SpecDec'] = sum(times)/len(times)
a2 = sum(accs)/len(accs) if accs else 0
print('  VisionCache+SpecDec:    %.1f ms | %.1f Hz | accept=%.0f%% | vis_skip=%d/%d' % 
      (results['VisionCache+SpecDec']*1000, 1/results['VisionCache+SpecDec'], a2/6*100, vis_skipped, N))

# Summary
base = results['Vanilla']
print('\n' + '=' * 70)
print('FINAL RESULTS')
print('=' * 70)
print('%-26s %10s %8s %9s' % ('Config', 'Latency', 'Freq', 'Speedup'))
print('-' * 56)
for name, t in results.items():
    print('%-26s %8.1f ms %6.1f Hz %7.2fx' % (name, t*1000, 1/t, base/t))
print('-' * 56)
print('Vision cache hit: %d/%d frames (%.0f%%)' % (vis_skipped, N, vis_skipped/N*100))
print('=' * 70)
