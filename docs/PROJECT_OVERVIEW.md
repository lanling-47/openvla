# 项目全景：VLA 模型推理加速系统

## 一句话总结

从零手写 LLM 推理引擎 (nano-vllm)，设计 Temporal Speculative Decoding 加速 VLA 模型，在 A100 上实测将 OpenVLA 7B 从 2.8Hz 加速到 12.5Hz（4.4x），并在 SmolVLA 450M 上验证 Vision Cache + Temporal Warm-Start 达到 48Hz。

---

## 项目架构

```
┌─────────────────────────────────────────────────────────────────────┐
│ 层级一：nano-vllm — 从零手写的 LLM 推理引擎                          │
│   • PagedAttention CUDA kernel (V1/V2, 959行 C++)                   │
│   • INT8 W8A16 Fused Matmul kernel                                  │
│   • FP8 KV Cache                                                    │
│   • Speculative Decoding 框架                                        │
│   • Roofline 分析 + 性能调优工具                                     │
├─────────────────────────────────────────────────────────────────────┤
│ 层级二：VLA-Spec — OpenVLA 7B 推理优化                               │
│   • Temporal Speculative Decoding (核心创新)                         │
│   • Vision Encoder Cache                                            │
│   • INT8 量化 + TensorRT 部署 Pipeline                              │
│   • OpenVLA 7B 端到端推理引擎                                        │
├─────────────────────────────────────────────────────────────────────┤
│ 层级三：SmolVLA 端侧优化 — 450M 模型 + Flow-Matching                │
│   • Temporal Warm-Start (时序去噪步数削减)                           │
│   • Vision Cache (跳过 VLM 编码)                                    │
│   • TensorRT INT8 部署 (Jetson Orin)                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 核心成果

### OpenVLA 7B (自回归模型, A100 实测)

| 方法 | 延迟 | 频率 | 加速比 |
|---|---|---|---|
| Vanilla FP16 | 353 ms | 2.8 Hz | 1.0x |
| + Temporal SpecDec | 183 ms | 5.5 Hz | 1.93x |
| + Vision Cache + SpecDec | 80 ms | 12.5 Hz | **4.4x** |
| 稳定帧 (100% accept) | 100 ms | 10 Hz | 2.5x |

### SmolVLA 450M (Flow-Matching 模型, A100 实测)

| 方法 | 延迟 | 频率 | 加速比 |
|---|---|---|---|
| Baseline (VLM + 10步去噪) | 525.5 ms | 1.9 Hz | 1.0x |
| + Vision Cache + Warm-Start (全帧) | 70.7 ms | 14.1 Hz | **7.4x** |
| + Vision Cache + Warm-Start (平滑帧) | 1.2 ms | 48.4 Hz | **25x** |

### nano-vllm PagedAttention Kernel (RTX 2080 Ti)

| 配置 | Custom Kernel | PyTorch SDPA+gather | 加速比 |
|---|---|---|---|
| B=64, Ctx=512 | 0.758 ms | 5.946 ms | **8x** |
| B=64, Ctx=2048 | 2.958 ms | 23.804 ms | **8x** |

---

## 技术亮点

### 1. Temporal Speculative Decoding (核心创新)

**问题**：标准投机解码需要额外的 draft model（训练成本 + 显存 + 计算）

**洞察**：机器人连续帧的动作高度相似（物理世界是连续的）

**方法**：把上一帧的 action tokens 直接当 draft → 零成本！

```python
# 算法流程
1. Prefill 当前帧 → 得到 first token
2. 拼接 [first] + prev_tokens[1:6] 作为 draft
3. Target LLM 一次性验证全部 6 个 draft tokens
4. 接受匹配的，不匹配时 fallback 到 vanilla decode
```

### 2. 手写 CUDA PagedAttention Kernel

- FlashAttention-style online softmax + 分页 KV cache
- V1→V2 优化：FP16 共享内存 + 向量化加载
- Occupancy 12% → 25%，带宽利用率 14% → 61%
- 支持 GQA + 多分区并行(长序列)

### 3. 跨架构泛化

同一个物理洞察（时序连续性），两种不同实现：
- **自回归模型 (OpenVLA)**：上一帧 tokens 做 draft → Temporal Speculative Decoding
- **Flow-Matching 模型 (SmolVLA)**：上一帧 action 做去噪起点 → Temporal Warm-Start

---

## 项目定位

| 角色 | 说明 |
|---|---|
| nano-vllm | **学习型项目**：从零理解推理引擎每一层设计决策 |
| VLA-Spec | **方法创新**：Temporal SpecDec 加速自回归 VLA |
| SmolVLA 优化 | **落地验证**：端侧部署的完整 pipeline |
| TensorRT (Orin) | **执行层**：单次 forward pass 的最优计算 |
| 系统级优化 | **调度层**：Vision Cache + Temporal + Async（TensorRT 管不了的） |

---

## 和 vLLM / TensorRT-LLM 的关系

> "nano-vllm 定位是从零理解推理引擎，不是对标 vLLM。
> 主要差距在 Tensor Core 和调度层——我的 kernel 是标量计算，带宽利用率 60%，FlashAttention-2 用 WMMA 能到 80%+。
> 但正因为从零写，我对 PagedAttention 的每一个设计决策（online softmax、block size 选择、occupancy/shared memory trade-off）都有深入理解。
> 在 Orin 部署时，执行层用 TensorRT，我的优化在其上层做系统级调度。"
