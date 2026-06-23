"""
nano-vllm: Lightweight LLM/VLA inference engine.

Core components:
  - kernels: PagedAttention (Triton), INT8 W8A16 fused matmul (CUDA)
  - cache: Paged KV Cache management
  - decoding: Speculative decoding (standard + temporal for VLA)
  - quantization: INT8 per-channel weight quantization
"""
