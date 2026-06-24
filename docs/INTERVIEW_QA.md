# 面试问答手册

## STAR 叙事框架（30 秒版）

> **Situation**: VLA 模型 OpenVLA 7B 推理 2.8Hz，远低于机器人实时控制 10Hz 要求
> **Task**: 加速到实时
> **Action**: Profile 定位瓶颈 (78% 在 decode) → 设计 Temporal SpecDec → 手写 CUDA kernel → 量化 → 端侧 SmolVLA 部署
> **Result**: 2.8Hz → 12.5Hz (4.4x)，SmolVLA 平滑帧 48Hz，突破实时阈值

---

## 高频面试题

### Q1: 你的 kernel 为什么选 128 threads per block？

**答**：因为 head_dim=128。我设计为每个线程负责输出向量的一个元素，所以刚好 128 线程 = 4 warps。这样在最后写 output 时每个线程直接写自己的元素，无需额外 reduction。如果用 32 线程，每个线程要负责 4 个元素，V 加权累加时需要更复杂的 loop；如果用 256 线程，多出的 128 线程在 score 计算阶段（只需 TILE_SIZE=32 线程）完全空闲，浪费资源。

---

### Q2: V1 → V2 做了什么优化？为什么有效？

**答**：两个核心改动：
1. **FP16 共享内存**：K/V tile 从 float32 改为 half 存储 → 共享内存从 33KB 降到 17KB → 一个 SM 能放 2 个 block（之前只能放 1 个）→ occupancy 从 12% 提升到 25%，更多 warp 可以 hide memory latency。
2. **向量化加载**：用 `uint32` 一次读 2 个 FP16 → 减少内存指令数 50% → 带宽利用率从 14% 提升到 61%。

计算时仍转 FP32 做乘加，所以精度不受影响。

---

### Q3: Online Softmax 和标准 Softmax 有什么区别？为什么必须用？

**答**：标准 softmax 需要两遍扫描：
1. 第一遍求全局 max
2. 第二遍算 exp(x-max)/sum

但 KV cache 是 memory-bound 的瓶颈，**多读一遍 KV 直接导致 2x 延迟**。

Online softmax 维护 running_max 和 running_sum，每处理一个 tile 就更新：
```
new_max = max(running_max, tile_max)
rescale = exp(running_max - new_max)
running_sum = running_sum * rescale + tile_sum
acc = acc * rescale + v_sum
```
只读一遍 KV，代价仅是多维护一个 rescale factor（几条指令）。

---

### Q4: 你的 kernel 和 FlashAttention 的差距在哪？

**答**：主要差距：
1. **没用 Tensor Core**：我用标量 FMA 做点积，FlashAttention-2 用 WMMA 指令做矩阵 tile 运算，算力差 ~10x
2. **带宽利用率**：我 60%，FA2 能到 80%+（更好的 double buffering + async copy）
3. **适用范围**：FA2 同时优化 prefill 和 decode，我只做了 decode

但 decode 阶段本身是 memory-bound（AI≈1.0），**Tensor Core 的算力优势在 decode 时发挥不大**。我的 kernel 重点优化的是带宽利用率，这个方向是正确的。

如果继续优化：① 引入 cp.async 做 double buffering ② 加 WMMA 做 score 计算 ③ 实现 split-K 减少 tail effect。

---

### Q5: Speculative Decoding 为什么能保证结果不变？

**答**：因为验证用的是 target model 本身。对于贪心解码：
- 如果 draft token == target 预测 → accept（和逐个 decode 结果完全一致）
- 如果 draft token != target 预测 → reject，用 target 的 token（也和逐个 decode 一致）

所以最终输出的每一个 token 都是 target model 的贪心选择，**数学上等价**。

---

### Q6: Temporal SpecDec 的核心洞察是什么？适用条件？

**答**：
- **洞察**：物理世界是连续的。机器人 20Hz 相机下，相邻帧动作变化极小。
- **适用条件**：动作时序连续的控制场景（机器人、自动驾驶）
- **不适用**：动作突变频繁的场景、通用 NLP 生成

实测平均 52% 接受率：匀速运动 100%，动作突变 0%（自动 fallback 到 vanilla decode，不影响正确性）。

---

### Q7: W8A16 量化为什么 decode 阶段用 FP16 而不是 INT8？

**答**：Decode 阶段 batch=1（M=1），做一次 INT8 量化转换的开销（计算 x_scale、round、clamp）可能比直接用 FP16 矩阵乘还慢。所以我缓存一份 FP16 权重，decode 时直接用——相当于把量化转换的开销分摊到了 prefill 阶段（那时 M 大，cuBLAS INT8 GEMM 的 Tensor Core 加速能补偿转换开销）。

这是一个 **hybrid 策略**：大 M 走 INT8 Tensor Core，小 M 走 FP16 cuBLAS。

---

### Q8: 如果接受率是 0%，你的方法会比 vanilla 慢吗？

**答**：会，但很少。最差情况（所有 draft 都被 reject）：
- 多做了 1 次 batched forward（verify pass）白做
- 额外延迟约 50ms（一次 prefill-like forward）
- 但因为 verify 得到了 first token 的正确 prediction，所以实际只浪费 1 次 forward

实测：动作突变帧约 1.0x（几乎不慢），因为 verify pass 的计算和 vanilla 的 first decode 基本等价。

---

### Q9: 为什么不直接用 vLLM？

**答**：
> "nano-vllm 定位是从零理解推理引擎，不是对标 vLLM。我需要理解 PagedAttention 内部每一个设计决策（online softmax、tile size、occupancy trade-off），才能正确设计上层系统优化（Vision Cache、Temporal SpecDec）。
>
> 在 Orin 部署时，执行层用 TensorRT（比 vLLM 更适合嵌入式），系统级优化是我在 TensorRT 之上做的一层调度逻辑——这些是 TensorRT 管不了的事。"

---

### Q10: SmolVLA 为什么瓶颈在 VLM 编码而不是去噪？

**答**：SmolVLM2 有 507M 参数、处理 256x256 图片生成 64 个 visual tokens，一次完整前向 69ms。而 Action Expert（Flow-Matching Transformer）很轻量（~200M），10 步去噪只要 2.2ms。

所以优化重点完全不同于 OpenVLA：
- OpenVLA 7B：瓶颈在 LLM decode（78%） → 用 SpecDec 减少 decode 次数
- SmolVLA 450M：瓶颈在 VLM 编码（99%） → 用 Vision Cache 跳过编码

这也说明**先 profile 再优化**的重要性——不同模型架构瓶颈完全不同。

---

### Q11: 如果动作突变频繁，你怎么改进？

**答**：几个方向：
1. **自适应 K**：根据历史 accept rate 动态调整 draft 长度（稳定时 K=6，不稳定时 K=2）
2. **轻量 predictor**：加一个极简的 motion predictor（如线性外推）做 draft，而非纯复制上一帧
3. **Confidence routing**：用 target 的 entropy 判断是否该做 SpecDec，entropy 高时直接走 vanilla

---

### Q12: 这个项目对你最大的 takeaway 是什么？

**答**：
1. **Profile 驱动**：不要猜瓶颈在哪，先 profile 再优化
2. **领域特化**：通用方法（标准 SpecDec）在特定领域可能不 work，但领域结构（时间连续性）可以设计出更好的方案
3. **分层思维**：清楚什么该自己写（系统级调度）、什么该用现成工具（TensorRT）
4. **从底层理解**：只有亲手写过 kernel，才能在面对性能问题时知道该往哪里看
