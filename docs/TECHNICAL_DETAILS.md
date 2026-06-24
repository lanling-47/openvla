# 技术点详解

## 1. PagedAttention CUDA Kernel

### 1.1 动机
KV cache 预分配导致大量内存碎片。类似 OS 虚拟内存，把 KV cache 分成固定大小的 block（页），通过 block_table 间接寻址。

### 1.2 核心设计
- **一个 thread block 处理一个 (batch, head) 对**
- **128 threads/block**：每线程负责 head_dim 中的一个元素
- **TILE_SIZE=32**：每次处理 32 个 KV tokens
- **Online Softmax**：O(1) 额外内存，一遍扫描完成 softmax

### 1.3 V1 → V2 优化
| 优化 | V1 | V2 | 效果 |
|---|---|---|---|
| 共享内存 | float32 (33KB) | FP16 (17KB) | 2 blocks/SM |
| Occupancy | 12% | 25% | +108% |
| 加载方式 | 逐元素 | uint32 向量化 | 带宽 14%→61% |

### 1.4 关键代码 (paged_attention.cu)
```cpp
// V2: 向量化 32-bit 加载 (2 x FP16 per instruction)
*reinterpret_cast<uint32_t*>(&smem->k_tile[token][dim]) =
    *reinterpret_cast<const uint32_t*>(&k_cache[offset]);

// Online softmax: 维护 running_max 和 running_sum
float new_max = fmaxf(running_max, tile_max);
float rescale = expf(running_max - new_max);
running_sum = running_sum * rescale + tile_sum;
acc = acc * rescale + v_sum;
```

---

## 2. Temporal Speculative Decoding

### 2.1 标准投机解码 vs 时序投机解码

| | 标准 Speculative Decoding | Temporal Speculative Decoding |
|---|---|---|
| Draft 来源 | 小模型 (需训练) | 上一帧 action tokens |
| 额外成本 | 显存 + 计算 + 训练 | **零** |
| 适用场景 | 通用 LLM | 时序连续场景 (机器人) |

### 2.2 算法
```python
def temporal_spec_decode(llm, input_embeds, prev_tokens, n=7):
    # 1. Prefill → first token
    out = llm(inputs_embeds=input_embeds, use_cache=True)
    first = out.logits[:, -1, :].argmax()

    # 2. 用上一帧 tokens 做 draft
    draft = prev_tokens[1:n]

    # 3. ONE batched forward pass 验证全部 6 个 draft
    verify_input = cat([first] + draft)
    t_out = llm(verify_input, past_key_values=kv)

    # 4. Accept/Reject
    for i in range(n-1):
        target = t_out.logits[:, i, :].argmax()
        if target == draft[i]:
            accept(target)
        else:
            reject(target)  # 用 target 的 token, fallback
            break
```

### 2.3 为什么 Layer-Pruned Draft 不行
Action tokens 集中在 vocab 31744-31999（256 个 bin），裁剪模型无法正确预测这个极窄的 token 范围。时序 draft 利用领域结构（物理连续性）而非模型结构。

---

## 3. INT8 W8A16 量化

### 3.1 Per-channel Absmax 量化
```python
# 量化
scale[n] = max(abs(W[n, :])) / 127
W_int8[n, k] = round(W[n, k] / scale[n]).clamp(-128, 127)

# 反量化 (fused into GEMM)
W_fp16[n, k] = W_int8[n, k] * scale[n]
```

### 3.2 Hybrid Prefill/Decode 策略
- **Prefill (M >= 16)**：cuBLAS INT8 Tensor Core GEMM → 1.66x 加速
- **Decode (M < 16)**：缓存 FP16 权重直接用 → 避免量化转换开销

### 3.3 效果
- LLM 权重 13.5GB → 7.0GB（48% 压缩）
- A100 上速度持平（FP16 已经很快）
- 核心价值：为 Jetson Orin TensorRT INT8 部署准备

---

## 4. SmolVLA Vision Cache + Temporal Warm-Start

### 4.1 Vision Cache
```python
# 连续帧像素差异很小 → 跳过 VLM 编码
diff = abs(current_frame - prev_frame).mean()
if diff < threshold:
    features = cached_features  # 省 523ms！
```
- 命中率：87% (26/30 帧)
- 效果：525ms → 1.2ms (跳过时)

### 4.2 Temporal Warm-Start (Flow-Matching 版本的时序优化)
```python
# 标准：从纯噪声开始，10步去噪
x = randn(...)
for t in range(10): x = denoise_step(x, t)

# Warm-Start：从上一帧 action 开始，只需3步
x = prev_action * 0.7 + noise * 0.3
for t in range(3): x = denoise_step(x, t)
```
- 10步 → 3步：去噪部分 **3.1x 加速**

### 4.3 同一洞察的双重实现
- 自回归 (OpenVLA): prev tokens → draft → **Speculative Decoding**
- Flow-Matching (SmolVLA): prev action → warm start → **Fewer Steps**

---

## 5. Roofline 分析与性能调优

### 5.1 Arithmetic Intensity
```
PagedAttention Decode:
  FLOPs = ctx_len × head_dim × 4 (Q@K + P@V)
  Bytes = ctx_len × head_dim × 4 (read K + V)
  AI ≈ 1.0 FLOP/byte → Memory-Bound!
```

### 5.2 优化方向
- Memory-bound → 优化带宽利用率（向量化加载、减少 bank conflict）
- 不需要 Tensor Core（compute-bound 才需要）
- V2 kernel: 带宽利用率 57-61%（理论峰值 616 GB/s）

---

## 6. 端侧部署分层

```
┌─────────────────────────────────────────┐
│ 系统级优化（你的贡献）                    │
│  • Vision Cache: 要不要跑 VLM?           │
│  • Temporal Warm-Start: 几步去噪?        │
│  • Async Pipeline: 何时触发推理?         │
├─────────────────────────────────────────┤
│ TensorRT Engine（NVIDIA 提供）           │
│  • INT8 量化推理                         │
│  • Layer fusion                         │
│  • sm_87 (Orin) kernel 选择             │
└─────────────────────────────────────────┘
```
