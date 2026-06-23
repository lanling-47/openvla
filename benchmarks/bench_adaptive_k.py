import torch, time, numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

print('=' * 70)
print('OpenVLA 7B — Adaptive K + Action Chunking + Profiling')
print('GPU:', torch.cuda.get_device_name(0))
print('=' * 70)

processor = AutoProcessor.from_pretrained('/root/models/openvla-7b', trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    '/root/models/openvla-7b', torch_dtype=torch.float16,
    low_cpu_mem_usage=True, trust_remote_code=True,
    attn_implementation='flash_attention_2',
).to('cuda:0').eval()
llm = model.language_model
print('Loaded: %.2f GB' % (torch.cuda.memory_allocated()/1e9))

prompt = 'In: What action should the robot take to pick up the red cup?\nOut:'
np.random.seed(42)
base_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

def make_frame(b, noise=5):
    return np.clip(b.astype(np.int16)+np.random.randint(-noise,noise+1,b.shape,dtype=np.int16),0,255).astype(np.uint8)

@torch.inference_mode()
def vision_encode(img):
    inputs = processor(prompt, Image.fromarray(img)).to('cuda:0', dtype=torch.float16)
    pf = model.vision_backbone(inputs['pixel_values'])
    proj = model.projector(pf)
    te = llm.get_input_embeddings()(inputs['input_ids'])
    return torch.cat([proj, te], dim=1)

@torch.inference_mode()
def spec_decode_k(emb, prev, k=6, n=7):
    # Speculative decode with variable K
    out = llm(inputs_embeds=emb, use_cache=True)
    kv = out.past_key_values
    first = out.logits[:,-1,:].argmax(dim=-1, keepdim=True)
    gen = [first.item()]; tc = 1; acc = 0
    if prev and len(prev) >= n and k > 0:
        rem = min(k, n-1)
        draft = [torch.tensor([[t]], device='cuda:0') for t in prev[1:rem+1]]
        verify = torch.cat([first]+draft, dim=1)
        t_out = llm(verify, past_key_values=kv, use_cache=True)
        kv = t_out.past_key_values; tc += 1
        last = first
        for i in range(rem):
            tp = t_out.logits[:,i,:].argmax(dim=-1).item()
            if tp == draft[i].item():
                acc += 1; gen.append(tp); last = draft[i]
                if len(gen) >= n: break
            else:
                gen.append(tp); last = torch.tensor([[tp]], device='cuda:0'); break
        if acc == rem and len(gen) < n:
            bonus = t_out.logits[:,rem,:].argmax(dim=-1).item()
            gen.append(bonus); last = torch.tensor([[bonus]], device='cuda:0')
        while len(gen) < n:
            t_out = llm(last, past_key_values=kv, use_cache=True)
            kv = t_out.past_key_values
            tok = t_out.logits[:,-1,:].argmax(dim=-1, keepdim=True)
            gen.append(tok.item()); last = tok; tc += 1
    else:
        tok = first
        for _ in range(n-1):
            out = llm(tok, past_key_values=kv, use_cache=True)
            kv = out.past_key_values
            tok = out.logits[:,-1,:].argmax(dim=-1, keepdim=True)
            gen.append(tok.item()); tc += 1
    return gen, tc, acc

# Warmup
e = vision_encode(base_img)
for _ in range(3): spec_decode_k(e, None)
torch.cuda.synchronize()

N = 30

# ====== PART 1: Adaptive K ======
print('\n' + '='*70)
print('PART 1: Adaptive K Strategy')
print('='*70)

np.random.seed(42); cur = base_img.copy()
times = []; accs = []; ks_used = []
prev = None; cached_emb = None; prev_frame = None; history = []

for f in range(N):
    if f > 0: cur = make_frame(cur, 5)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    if prev_frame is not None:
        diff = np.abs(cur.astype(np.int16) - prev_frame.astype(np.int16)).mean()
        if diff < 5.0: emb = cached_emb
        else: emb = vision_encode(cur); cached_emb = emb
    else: emb = vision_encode(cur); cached_emb = emb
    prev_frame = cur.copy()
    # Adaptive K rule
    if len(history) >= 2 and all(h >= 5 for h in history[-2:]):
        k = 6
    elif len(history) >= 1 and history[-1] == 0:
        k = 3
    else:
        k = 4
    toks, tc, acc = spec_decode_k(emb, prev, k=k)
    torch.cuda.synchronize(); times.append(time.perf_counter()-t0)
    history.append(acc); ks_used.append(k)
    if f > 0: accs.append(acc)
    prev = toks

adaptive_avg = sum(times)/len(times)
print('  Adaptive: %.1f ms | %.1f Hz | avg_k=%.1f' % (adaptive_avg*1000, 1/adaptive_avg, sum(ks_used)/len(ks_used)))
print('  K dist: %s' % {k: ks_used.count(k) for k in sorted(set(ks_used))})

# Fixed K=6
np.random.seed(42); cur = base_img.copy()
times2 = []; prev = None; cached_emb = None; prev_frame = None
for f in range(N):
    if f > 0: cur = make_frame(cur, 5)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    if prev_frame is not None:
        diff = np.abs(cur.astype(np.int16) - prev_frame.astype(np.int16)).mean()
        if diff < 5.0: emb = cached_emb
        else: emb = vision_encode(cur); cached_emb = emb
    else: emb = vision_encode(cur); cached_emb = emb
    prev_frame = cur.copy()
    toks, tc, acc = spec_decode_k(emb, prev, k=6)
    torch.cuda.synchronize(); times2.append(time.perf_counter()-t0)
    prev = toks
fixed_avg = sum(times2)/len(times2)
print('  Fixed K=6: %.1f ms | %.1f Hz' % (fixed_avg*1000, 1/fixed_avg))

# ====== PART 2: Action Chunking ======
print('\n' + '='*70)
print('PART 2: Action Chunking')
print('='*70)

np.random.seed(42)
for chunk in [1, 2, 4]:
    cur = base_img.copy()
    n_tok = 7 * chunk
    times_c = []; prev = None; cached_emb = None; prev_frame = None
    for f in range(N):
        if f > 0: cur = make_frame(cur, 5)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        if prev_frame is not None:
            diff = np.abs(cur.astype(np.int16) - prev_frame.astype(np.int16)).mean()
            if diff < 5.0: emb = cached_emb
            else: emb = vision_encode(cur); cached_emb = emb
        else: emb = vision_encode(cur); cached_emb = emb
        prev_frame = cur.copy()
        toks, tc, acc = spec_decode_k(emb, prev, k=6, n=n_tok)
        torch.cuda.synchronize(); times_c.append(time.perf_counter()-t0)
        prev = toks
    avg_c = sum(times_c)/len(times_c)
    eff = chunk / avg_c
    print('  Chunk=%d (%2d tok): %.1f ms | %.1f Hz infer | %.1f actions/s' % (chunk, n_tok, avg_c*1000, 1/avg_c, eff))

# ====== PART 3: Profiling ======
print('\n' + '='*70)
print('PART 3: Latency Breakdown')
print('='*70)

vis_t, pre_t, dec_t, spec_t = [], [], [], []
for _ in range(10):
    cur = make_frame(base_img, 5)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    emb = vision_encode(cur)
    torch.cuda.synchronize(); vis_t.append(time.perf_counter()-t0)

    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = llm(inputs_embeds=emb, use_cache=True)
    torch.cuda.synchronize(); pre_t.append(time.perf_counter()-t0)

    kv = out.past_key_values
    tok = out.logits[:,-1,:].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(6):
        out = llm(tok, past_key_values=kv, use_cache=True); kv = out.past_key_values
        tok = out.logits[:,-1,:].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize(); dec_t.append(time.perf_counter()-t0)

    torch.cuda.synchronize(); t0 = time.perf_counter()
    dummy = torch.cat([tok]+[torch.tensor([[31865]],device='cuda:0') for _ in range(6)], dim=1)
    _ = llm(dummy, past_key_values=kv, use_cache=True)
    torch.cuda.synchronize(); spec_t.append(time.perf_counter()-t0)

v = sum(vis_t)/len(vis_t)*1000
p = sum(pre_t)/len(pre_t)*1000
d = sum(dec_t)/len(dec_t)*1000
s = sum(spec_t)/len(spec_t)*1000
total = v+p+d

print('  Vision Encode:   %6.1f ms (%4.1f%%)' % (v, v/total*100))
print('  LLM Prefill:     %6.1f ms (%4.1f%%)' % (p, p/total*100))
print('  LLM Decode x6:   %6.1f ms (%4.1f%%)' % (d, d/total*100))
print('  Spec Verify x1:  %6.1f ms' % s)
print('  ' + '-'*40)
print('  Vanilla total:   %6.1f ms' % total)
print('  +SpecDec:        %6.1f ms (save %.1f ms)' % (v+p+s, d-s))
print('  +Cache+Spec:     %6.1f ms (save %.1f ms)' % (p+s, v+d-s))
print('  Savings: Vision=%.0f%%, Decode=%.0f%%' % (v/total*100, (d-s)/total*100))
print('=' * 70)
