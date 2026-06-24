# VLA 推理加速全栈实践：从手写 CUDA Kernel 到端侧 48Hz 实时控制

> 本文完整记录了一个 VLA（Vision-Language-Action）模型推理加速项目的全过程。从零手写 LLM 推理引擎（nano-vllm），到设计 Temporal Speculative Decoding 将 OpenVLA 7B 从 2.8Hz 加速到 12.5Hz，再到 SmolVLA 450M 端侧优化实现 48Hz 实时控制——覆盖底层 CUDA kernel、系统级优化、跨架构泛化三个层次。

---

## 一、项目全景

```
┌──────────────────────────────────────────────────────────────────┐
│ 底层引擎：nano-vllm (手写 CUDA)                                   │
│   PagedAttention kernel · INT8 fused matmul · FP8 KV cache       │
├──────────────────────────────────────────────────────────────────┤
│ 方法创新：VLA-Spec (OpenVLA 7B, 自回归)                           │
│   Temporal Speculative Decoding · Vision Cache · INT8 量化        │
├──────────────────────────────────────────────────────────────────┤
│ 落地验证：SmolVLA 450M (Flow-Matching)                            │
│   Temporal Warm-Start · Vision Cache · TensorRT 部署              │
└──────────────────────────────────────────────────────────────────┘
```

核心理念：**同一个物理洞察（动作时序连续性），在不同模型架构下设计不同优化方案**。

---

## 二、nano-vllm：从零手写 LLM 推理引擎

### 2.1 为什么要从零写？

不是为了替代 vLLM，而是为了**深入理解推理引擎每一层的设计决策**。只有亲手实现过 PagedAttention，才能在设计上层系统优化时做出正确的工程判断。

### 2.2 PagedAttention CUDA Kernel

**核心设计**：FlashAttention-style online softmax + 分页 KV cache 间接寻址。

```cpp
// 一个 thread block 处理一个 (batch, head) 对
// 128 threads = 4 warps, 每线程负责 output 的一个 head_dim 元素
// 每次处理 TILE_SIZE=32 个 KV tokens

// Online softmax: 一遍扫描，O(1) 额外内存
float new_max = fmaxf(running_max, tile_max);
float rescale = expf(running_max - new_max);
running_sum = running_sum * rescale + tile_sum;
acc = acc * rescale + v_sum;  // 累加加权 V
```

**V1 → V2 优化**：

| 维度 | V1 | V2 | 效果 |
|---|---|---|---|
| 共享内存 | K/V 存 float32 (33KB) | 存 FP16 (17KB) | 2 blocks/SM |
| Occupancy | 12% | **25%** | 更多 warp hide latency |
| 加载方式 | 逐元素 | uint32 向量化 (2×FP16/指令) | 带宽 14% → **61%** |
| 精度 | FP32 计算 | FP16 存储 + FP32 计算 | **无损** |

**性能数据** (RTX 2080 Ti)：

```
Batch=64, Ctx=512:   Custom 0.758ms vs SDPA+gather 5.946ms → 8x
Batch=64, Ctx=2048:  Custom 2.958ms vs SDPA+gather 23.804ms → 8x
带宽利用率: 57-61% (理论峰值 616 GB/s)
```

### 2.3 Roofline 分析

```
PagedAttention Decode:
  FLOPs ≈ ctx_len × head_dim × 4
  Bytes ≈ ctx_len × head_dim × 4
  Arithmetic Intensity ≈ 1.0 FLOP/byte → Memory-Bound!
```

定位为 memory-bound 后，优化方向明确：提升带宽利用率（向量化、减少 bank conflict），而非堆算力（Tensor Core）。

### 2.4 和 FlashAttention-2 的差距

| | nano-vllm kernel | FlashAttention-2 |
|---|---|---|
| 计算方式 | 标量 FMA | WMMA Tensor Core |
| 带宽利用率 | 60% | 80%+ |
| Double buffering | 无 | cp.async + 双缓冲 |
| 定位 | 学习型 + 设计依据 | 生产级 |

**关键认知**：Decode 阶段本身 memory-bound，Tensor Core 的算力优势发挥不大。我的 kernel 重点在带宽利用率，方向正确。

---

## 三、Temporal Speculative Decoding：核心创新

### 3.1 问题

OpenVLA 7B 生成 7 个 action tokens 需要 7 次 sequential LLM forward，78% 时间花在 decode。

```
OpenVLA 7B 延迟分解 (A100):
  Vision Encoder:   29.5 ms ( 7.7%)
  LLM Prefill:      54.4 ms (14.2%)
  LLM Decode ×6:   299.0 ms (78.1%) ← 瓶颈！
  Total:           383 ms = 2.6 Hz
```

### 3.2 标准投机解码为什么不行？

- **层裁剪失败**：Action tokens 集中在 vocab 31744-31999（极窄的 256-bin 空间），裁剪模型输出完全是 garbage
- **额外成本**：需要训练 draft model + 额外显存 + 额外计算

### 3.3 我们的方法：利用时间连续性

**核心洞察**：物理世界是连续的。20Hz 相机下，相邻帧的机械臂动作几乎相同。

```python
# 上一帧 action: [31865, 31864, 31874, 31874, 31864, 31855, 31744]
# 当前帧 action: [31865, 31864, 31874, 31874, 31864, 31855, 31744]
#                 ^^^^^^^ 匀速运动时完全一样！
```

**算法**：

```
1. Prefill 当前帧 [visual + text tokens] → 得到 first token
2. 拼接 [first_token] + prev_frame_tokens[1:6] 作为 draft
3. Target LLM 一次性验证 6 个 draft tokens (ONE batched forward pass)
4. 逐个比较：匹配则 accept，不匹配则 reject 并用 target 的 token
5. Fallback: 如果全部 reject，seamlessly 转入 vanilla decode
```

**关键优势**：
- **零成本 draft**：不需要额外模型、不需要训练、不占额外显存
- **数学等价**：贪心解码下输出与 vanilla 完全一致
- **优雅降级**：动作突变时自动 fallback，不影响正确性

### 3.4 A100 实测结果

```
Vanilla:              353 ms = 2.8 Hz (baseline)
+ Temporal SpecDec:   183 ms = 5.5 Hz (1.93x)
+ Vision Cache:        80 ms = 12.5 Hz (4.40x) ← 超越 10Hz!
```

**Per-frame 分析** (30 帧连续)：

| 阶段 | 帧 | 接受率 | 加速 | 行为 |
|---|---|---|---|---|
| 匀速运动 | 1-4 | 100% (6/6) | 2.5x | 全部 draft 接受 |
| 动作突变 | 5-6 | 0% | 1.0x | 自动 fallback |
| 恢复期 | 7-9 | 33% (2/6) | 1.3x | 逐渐收敛 |
| 稳态 | 10+ | ~50% | 1.5x | 混合 accept/reject |

---

## 四、INT8 量化：为端侧部署准备

### 4.1 Per-channel Absmax 量化

```python
# 对 Llama-2 7B 全部 224 个 Linear 层做量化
scale = w.abs().amax(dim=1) / 127.0
w_int8 = (w / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
```

### 4.2 Hybrid Prefill/Decode 策略

| 阶段 | 策略 | 原因 |
|---|---|---|
| Prefill (M≥16) | cuBLAS INT8 Tensor Core GEMM | M 大时量化转换开销可被 Tensor Core 加速补偿 |
| Decode (M=1) | 缓存 FP16 权重直接用 | M=1 时量化转换比 FP16 matmul 还慢 |

### 4.3 效果

- 权重内存：13.5GB → 7.0GB (**48% 压缩**)
- Prefill 单 GEMM：**1.66x 加速**
- 端到端速度：与 FP16 持平（A100 上 FP16 已极快）
- **核心价值**：为 Jetson Orin TensorRT INT8 部署提供量化权重

---

## 五、SmolVLA 450M：端侧优化与跨架构验证

### 5.1 为什么选 SmolVLA？

| 对比 | OpenVLA 7B | SmolVLA 450M |
|---|---|---|
| 参数量 | 7B | 450M (16x 更小) |
| 动作生成 | 自回归 token-by-token | Flow-Matching 迭代去噪 |
| 硬件需求 | A100 40GB | **Jetson Orin Nano 8GB** |
| 社区生态 | 一般 | HuggingFace + LeRobot |

### 5.2 Profile：瓶颈完全不同

```
SmolVLA 延迟分解 (A100):
  VLM Backbone (SmolVLM2):  523.0 ms (99.5%) ← 新瓶颈！
  Action Expert (10步去噪):    2.5 ms ( 0.5%)
  Total:                    525.5 ms = 1.9 Hz
```

**惊人发现**：SmolVLA 的瓶颈是 VLM 编码（99.5%），不是去噪！这与 OpenVLA 完全相反。

### 5.3 Vision Cache：最大加速来源

```python
# 连续帧像素差异极小 → 跳过 VLM 编码
img_diff = abs(current - previous).mean()
if img_diff < 10.0:  # pixel threshold
    features = cached_features  # 省 523ms！
```

- 命中率：**87%** (26/30 帧)
- 命中时延迟：**1.2ms**（只做 3 步去噪）
- 未命中时延迟：**525ms**（全量 VLM + 10 步去噪）

### 5.4 Temporal Warm-Start：同一洞察的 Flow-Matching 版本

```python
# 标准 Flow-Matching: 从纯噪声开始，10步 Euler 积分
x = torch.randn(1, chunk_size, action_dim)
for t in linspace(0, 1, steps=10):
    x = x + velocity_net(x, features, t) * dt

# Temporal Warm-Start: 从上一帧 action 开始，只需3步
x = prev_action * 0.7 + noise * 0.3  # 接近目标的起点
for t in linspace(0.7, 1, steps=3):    # 只走最后30%路径
    x = x + velocity_net(x, features, t) * dt
```

去噪加速：10 步 2.18ms → 3 步 0.70ms = **3.1x**

### 5.5 A100 实测结果

| 方法 | 延迟 | 频率 | 加速比 |
|---|---|---|---|
| Baseline (全量 VLM + 10步) | 525.5 ms | 1.9 Hz | 1.0x |
| **Optimized (全帧平均)** | **70.7 ms** | **14.1 Hz** | **7.4x** |
| **Optimized (平滑帧)** | **1.2 ms** | **48.4 Hz** | **25x** |
| 动作突变帧 (fallback) | 521.5 ms | 1.9 Hz | 回退 |

### 5.6 跨架构泛化的意义

**同一个物理洞察 → 两种不同优化**：

| 模型架构 | 瓶颈 | 时序优化方案 |
|---|---|---|
| OpenVLA (自回归) | LLM decode 78% | 上一帧 tokens 做 draft → SpecDec |
| SmolVLA (Flow-Matching) | VLM 编码 99.5% | 上一帧 action 做去噪起点 → Warm-Start |

这说明理解的是**原理**（时序连续性），而非**套路**（只会做 SpecDec）。

---

## 六、端侧部署架构

### 6.1 分层设计

```
┌───────────────────────────────────────────────────────────┐
│ 系统级优化（我的贡献，TensorRT 管不了的事）                  │
│  • Vision Cache: 这帧该不该跑 VLM？                        │
│  • Temporal Warm-Start: 从哪个起点开始去噪？几步？          │
│  • Async Pipeline: 何时触发新一轮推理？                     │
│  • Fallback: 检测到动作突变时自动切回全量推理               │
├───────────────────────────────────────────────────────────┤
│ TensorRT Engine（NVIDIA 提供，单次 forward 的最优执行）     │
│  • INT8 量化推理 (Tensor Core)                             │
│  • Layer fusion (减少 kernel launch)                       │
│  • sm_87 (Orin) 最优 kernel 选择                           │
└───────────────────────────────────────────────────────────┘
```

### 6.2 为什么 nano-vllm 不在 Orin 上跑？

在 Orin 上做单模型推理，TensorRT 是最优选择（NVIDIA 官方为 Jetson 深度优化）。nano-vllm 的价值是：
- A100 上做方法验证（SpecDec 效果验证）
- 理解底层原理（设计系统级优化的依据）
- 面试时证明底层能力

---

## 七、工程方法论

### 7.1 Profile → 定位瓶颈 → 设计优化

```
OpenVLA:  profile → 78% 在 decode → 减少 decode 次数 (SpecDec)
SmolVLA:  profile → 99% 在 VLM 编码 → 跳过冗余编码 (Cache)
```

**不同模型瓶颈完全不同**，盲目套用优化方法会南辕北辙。

### 7.2 正交优化逐层叠加

每个优化独立生效、互不干扰：
- Temporal SpecDec 减少 LLM 调用次数 ⟂ Vision Cache 跳过视觉编码
- INT8 量化压缩权重 ⟂ PagedAttention 管理 KV cache

### 7.3 优雅降级

所有优化在"最差情况"下不劣于 baseline：
- SpecDec 全 reject → 等价 vanilla decode（+1 次 verify 开销 ≈ 0）
- Vision Cache 未命中 → 正常编码
- Warm-Start 效果差 → fallback 到 10 步去噪

---

## 八、总结

| 指标 | OpenVLA 优化前 | OpenVLA 优化后 | SmolVLA 优化后 |
|---|---|---|---|
| 延迟 | 353 ms | 80 ms | 1.2 ms (平滑帧) |
| 频率 | 2.8 Hz | 12.5 Hz | 48.4 Hz |
| 加速比 | - | 4.4x | 25x |
| 权重显存 | 13.5 GB | 7.0 GB | 0.9 GB |
| 实时阈值 (10Hz) | 远低于 | **超越** | **远超** |

**方法论总结**：

```
先 profile 定位瓶颈 → 设计场景特化优化 → 逐层叠加正交优化
→ 跨架构验证泛化性 → 设计端侧部署 pipeline
```

---

## 相关链接

- 项目代码：https://github.com/lanling-47/openvla
- nano-vllm CUDA kernel：https://github.com/lanling-47/nano-vllm
- SmolVLA 原始模型：https://huggingface.co/lerobot/smolvla_base
- 参考论文：
  - [PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180)
  - [FlashAttention-2](https://arxiv.org/abs/2307.08691)
  - [Speculative Decoding](https://arxiv.org/abs/2211.17192)
  - [OpenVLA](https://arxiv.org/abs/2406.09246)
  - [SmolVLA](https://arxiv.org/abs/2506.01844)
