# 从零到一：nano-vllm + VLA-Spec 项目学习路线图

## 前置说明

- 目标：让一个有 Python/C++ 基础但不了解 CUDA/LLM 推理的同学，在 4-6 周内学会这个项目的所有技术点
- 每个知识点标注了【面试频率】：高频 / 中频 / 低频
- 每个阶段有【检验标准】：能回答的面试问题

---

## 阶段一：GPU 编程基础（第 1 周）

### 1.1 CUDA 编程模型 【高频】
- 概念：Grid → Block → Thread 的层级关系
- 关键词：threadIdx, blockIdx, blockDim, __syncthreads()
- 学习资源：CUDA C Programming Guide Chapter 1-3
- 检验：能解释"128 threads per block, 4 warps"是什么意思

### 1.2 GPU 内存层次 【高频】
- 全局内存 (Global Memory): 大但慢，~600 GB/s（RTX 2080 Ti）
- 共享内存 (Shared Memory): 小但快，48KB/SM，~10 TB/s
- 寄存器 (Registers): 最快，每线程私有
- 检验：能画出 GPU 内存层次图并标注带宽

### 1.3 Warp 和 SIMT 执行模型 【高频】
- 32 个线程组成一个 warp，同步执行同一条指令
- Warp divergence 的性能影响
- Warp-level primitives: __shfl_xor_sync（用于 reduction）
- 检验：能解释"warp_reduce_max"为什么用 __shfl_xor_sync 而不是 shared memory

### 1.4 Occupancy 和性能分析 【中频】
- Occupancy = 活跃 warp 数 / SM 最大 warp 数
- 影响因素：shared memory 使用量、寄存器数、block 大小
- Roofline 模型：compute-bound vs memory-bound
- 检验：能解释 V1 kernel 12% occupancy → V2 25% occupancy 的原因

---

## 阶段二：Transformer 和 LLM 推理基础（第 1-2 周）

### 2.1 Self-Attention 机制 【高频】
- Q, K, V 的含义和计算：Attention(Q,K,V) = softmax(QK^T / sqrt(d)) V
- 时间复杂度：O(n^2 d)
- Multi-Head Attention：多个 head 并行计算
- 检验：能手写 attention 的 PyTorch 代码

### 2.2 KV Cache 【高频】
- 为什么需要 KV Cache：避免自回归时重复计算前面 token 的 K/V
- 内存占用：2 × num_layers × seq_len × hidden_size × dtype_bytes
- 检验：能算出 Llama-2 7B 在 seq_len=2048 时 KV cache 占多少显存

### 2.3 Prefill vs Decode 阶段 【高频】
- Prefill：处理所有输入 token（compute-bound，大矩阵乘）
- Decode：每次只生成 1 个新 token（memory-bound，读 KV cache）
- 为什么 decode 是瓶颈：每生成 1 个 token 就要读一遍全部 KV cache
- 检验：能解释"78% 时间花在 decode"的原因

### 2.4 Grouped Query Attention (GQA) 【中频】
- MHA: num_q_heads = num_kv_heads（每个 Q head 有独立的 KV head）
- GQA: num_q_heads > num_kv_heads（多个 Q head 共享同一个 KV head）
- 优势：减少 KV cache 显存，不显著影响质量
- 检验：Qwen3-0.6B 的 16 Q heads / 8 KV heads 意味着 group_size=2

---

## 阶段三：PagedAttention 核心 kernel（第 2-3 周）

### 3.1 PagedAttention 的动机 【高频】
- 问题：KV cache 预分配导致巨大内存浪费（碎片化）
- 方案：类似操作系统虚拟内存，把 KV cache 分成固定大小的 block（页）
- block_table：sequence → physical block 的映射表
- 检验：能解释 block_table[batch_idx][blk_idx] 的含义

### 3.2 Online Softmax 算法 【高频】
- 问题：标准 softmax 需要两遍扫描（先求 max，再算 exp/sum）
- 方案：维护 running_max 和 running_sum，一遍扫描完成
- 核心公式：
  - new_max = max(running_max, tile_max)
  - rescale = exp(running_max - new_max)
  - running_sum = running_sum * rescale + tile_sum
  - acc = acc * rescale + new_weighted_v
- 检验：能在白板上推导 online softmax 的正确性

### 3.3 V1 Kernel 实现细节 【中频】
- 一个 thread block 处理一个 (batch, head) 对
- 128 threads：每个线程负责一个 head_dim 元素的输出
- TILE_SIZE=32：每次处理 32 个 KV token
- 共享内存布局：q[128] + k_tile[32][128] + v_tile[32][128] + scores[32] = 33KB
- 检验：能解释为什么用 128 个线程而不是 32 或 256

### 3.4 V2 优化：半精度共享内存 + 向量化加载 【高频】
- 优化 1：K/V tile 存 FP16 而非 FP32 → 共享内存 33KB → 17KB
  - 效果：1 block/SM → 2 blocks/SM，occupancy 12% → 25%
- 优化 2：uint32 向量化加载（一条指令读 2 个 FP16）→ 带宽利用率 14% → 61%
- 计算时仍转 FP32 → 无精度损失
- 检验：能解释 *reinterpret_cast<uint32_t*>(...) 在做什么

### 3.5 分区并行（Partitioned Attention）【低频】
- 问题：长序列时一个 block 串行处理所有 tile 太慢
- 方案：把序列按 partition_size=512 切分，每段独立计算，最后 reduce
- 实现：3D grid (batch, heads, partitions) + reduce_partitioned_attention_kernel
- 检验：能解释 partial_max 和 partial_sum 如何合并

---

## 阶段四：量化技术（第 3-4 周）

### 4.1 INT8 量化原理 【高频】
- Per-channel absmax 量化：
  - scale[n] = max(abs(W[n, :])) / 127
  - W_int8[n, k] = round(W[n, k] / scale[n])
- 反量化：W_fp16[n, k] = W_int8[n, k] * scale[n]
- 检验：能手算一个 4 元素向量的量化和反量化过程

### 4.2 W8A16 融合 matmul 的含义 【高频】
- W8A16：权重 INT8，激活 FP16
- "融合"：反量化不单独做一遍，而是在 GEMM 的 inner loop 里乘 scale
- 好处：不需要额外的 FP16 中间权重张量（省一半显存）
- 检验：能解释 quant_matmul.cu 中 w_tile[wn][wk] = (int8)w * scale 这行

### 4.3 cuBLAS INT8 GEMM (torch._int_mm) 【中频】
- Prefill 阶段 M 较大（batch × seq_len），适合用 Tensor Core INT8 GEMM
- Decode 阶段 M=1，不值得做量化转换，直接用缓存的 FP16 权重
- Hybrid 策略：M >= 16 用 INT8 GEMM，M < 16 用 FP16
- 检验：能解释为什么 decode 时不用 INT8 反而用 FP16

### 4.4 FP8 KV Cache 【低频】
- E4M3 格式：1 sign + 4 exponent (bias=7) + 3 mantissa
- 好处：KV cache 显存减半（FP16 → FP8）
- 实现：软件转换（Turing 没有 FP8 硬件），精度损失 < 0.001
- 检验：能解释 fp8_e4m3_to_float() 函数在做什么

---

## 阶段五：投机解码（第 4-5 周）

### 5.1 标准 Speculative Decoding 原理 【高频】
- 核心思想：小模型 (draft) 快速猜 K 个 token，大模型 (target) 一次性验证
- 验证规则：逐个比较，第一个不匹配就停，用 target 的 token 替换
- 加速原理：target 模型从调用 N 次变成 ~N/K 次
- 输出等价性：贪心解码下与 vanilla 完全一致（不改变结果）
- 检验：能画出 draft-verify-accept/reject 的流程图

### 5.2 Temporal Speculative Decoding（核心创新）【高频】
- 标准方法的问题：需要额外 draft model（显存、训练成本）
- 领域洞察：机器人连续帧动作高度相似（物理世界是连续的）
- 方法：把上一帧的 action tokens 直接当 draft（零成本！）
- 流程：
  1. Prefill 当前帧 → 得到 first token
  2. 拼接 [first] + prev_tokens[1:6] → 送入 target 一次验证
  3. 逐个比较 target 预测 vs draft，接受匹配的，不匹配就停
- 检验：能解释"零成本 draft"的含义和适用条件

### 5.3 Accept Rate 和性能关系 【中频】
- 全部接受（6/6）：2 次 LLM call → 2.5x 加速
- 完全不接受（0/6）：1 次 verify 白做 + fallback → ~1.0x（略慢）
- 平均 52% 接受率 → 4.2 次 call/frame → 1.5-1.65x 加速
- 什么时候接受率高：匀速运动、稳定帧
- 什么时候低：动作突变、新任务开始
- 检验：能从 bench_temporal_specdec.txt 的 per-frame 数据分析规律

### 5.4 和 Spec-VLA (EMNLP 2025) 的区别 【中频】
- Spec-VLA：训练一个轻量 draft model
- 本项目：利用时间连续性做 zero-cost draft
- 优势：不需要训练、不占额外显存、不增加计算
- 劣势：只适用于时序连续场景（机器人），通用场景不行
- 检验：能说清楚各自的 trade-off

---

## 阶段六：系统集成与面试准备（第 5-6 周）

### 6.1 OpenVLA 模型架构 【中频】
- 视觉编码器：DINOv2 + SigLIP → 256 visual tokens
- LLM 骨架：Llama-2 7B
- 动作解码：7 个 token，每个 token 是 256-bin 离散化的一个 DOF
- 检验：能画出完整的 Image → Action 数据流

### 6.2 Vision Encoder Caching 【低频】
- 洞察：20Hz 相机，50ms 内画面变化极小
- 方法：比较像素差异，低于阈值就复用上一帧的 visual embedding
- 效果：97% 命中率，省 29.5ms/帧
- 检验：能解释为什么这个优化是"正交"的

### 6.3 端侧部署 Pipeline 【低频】
- 路径：PyTorch → INT8 量化 → ONNX Export → TensorRT 编译 → Jetson 运行
- TensorRT 做什么：layer fusion、kernel 自动选择、INT8 calibration
- 预估 Jetson Orin 性能：~48ms = 20.8Hz
- 检验：能解释 ONNX 在部署中的角色

### 6.4 面试叙事结构 【高频】
推荐 STAR 框架讲述：
- **Situation**: VLA 模型 2.8Hz，远低于实时控制 10Hz 要求
- **Task**: 加速推理到实时
- **Action**: Profile 定位瓶颈(78% decode) → 设计 Temporal SpecDec → 手写 CUDA kernel → 量化
- **Result**: 2.8Hz → 12.5Hz，4.4x 加速，突破实时阈值

### 6.5 高频面试问题清单

**CUDA / 系统层面：**
1. 你的 kernel 为什么选 128 threads per block？
2. V1 → V2 做了什么优化？为什么有效？
3. Online softmax 和标准 softmax 有什么区别？为什么必须用？
4. 你的 kernel 和 FlashAttention 的差距在哪？下一步怎么优化？
5. 什么是 memory-bound？你怎么确认你的 kernel 是 memory-bound 的？

**推理优化层面：**
6. Speculative decoding 的原理？为什么能保证结果不变？
7. Temporal SpecDec 的核心洞察是什么？适用条件？
8. W8A16 量化为什么 decode 阶段用 FP16 而不是 INT8？
9. PagedAttention 解决了什么问题？block_table 怎么工作的？
10. 如果接受率是 0%，你的方法会比 vanilla 慢吗？慢多少？

**架构 / 设计层面：**
11. 为什么不直接用 vLLM？你的项目和 vLLM 的关系是什么？
12. 如果动作突变频繁，你怎么改进？
13. 端侧部署的瓶颈在哪？TensorRT 能解决什么？

---

## 学习资源推荐

| 阶段 | 资源 |
|---|---|
| CUDA 基础 | CUDA C Programming Guide (Chapters 1-5) |
| GPU 性能分析 | "Roofline Model" by Samuel Williams |
| Attention 机制 | "Attention Is All You Need" 原论文 |
| FlashAttention | FlashAttention-2 论文 + Tri Dao 的 blog |
| PagedAttention | vLLM 论文 "Efficient Memory Management for LLM Serving" |
| 投机解码 | "Fast Inference from Transformers via Speculative Decoding" (2022) |
| 量化 | "A Survey of Quantization Methods for Efficient Neural Network Inference" |

---

## 时间投入预估

| 阶段 | 时间 | 产出 |
|---|---|---|
| 阶段一 GPU 基础 | 5-7 天 | 能写简单 CUDA kernel（如矩阵加法、规约） |
| 阶段二 Transformer | 3-5 天 | 能手写 attention + KV cache 推理代码 |
| 阶段三 PagedAttention | 7-10 天 | 能逐行读懂 paged_attention.cu 并解释每个设计决策 |
| 阶段四 量化 | 3-5 天 | 能实现简单的 per-channel INT8 量化 |
| 阶段五 投机解码 | 5-7 天 | 能手写 speculative_decode 并解释 temporal 变体 |
| 阶段六 集成+面试 | 5-7 天 | 能流畅讲述完整项目故事 |
| **总计** | **4-6 周** | **面试可讲** |
