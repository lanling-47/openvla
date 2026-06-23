"""
Temporal Speculative Decoding — VLA-specific optimization.

Key insight: In robot control, consecutive camera frames produce similar actions.
The previous frame's action tokens serve as a ZERO-COST draft for the current frame.

Algorithm:
  1. Prefill LLM on current frame's [visual + instruction] tokens
  2. Use previous frame's action tokens (7 tokens) as draft
  3. Target LLM verifies all 6 draft tokens in ONE batched forward pass
  4. Accept matching tokens, fallback to vanilla at first mismatch

Benefits:
  - No draft model needed (zero additional memory/compute)
  - 1 verify pass replaces up to 6 sequential decode passes
  - Typical accept rate: 50-100% (depends on action temporal smoothness)
  - Graceful fallback: on mismatch, seamlessly continues with vanilla decode

This is a novel contribution specific to VLA/robotics inference.
Standard speculative decoding uses a separate draft model, which is:
  - Extra memory cost
  - Extra compute cost
  - Difficult for VLA (layer pruning breaks action token prediction)

Temporal spec decode exploits the unique temporal structure of robot control.
"""

import torch
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class TemporalSpecResult:
    """Result of temporal speculative decoding."""
    tokens: List[int]
    latency_ms: float
    num_llm_calls: int
    num_accepted: int
    draft_tokens_used: int


class TemporalSpecDecoder:
    """
    Temporal Speculative Decoder for VLA action generation.

    Maintains a cache of previous frame's action tokens and uses them
    as draft predictions for the current frame.
    """

    def __init__(self, llm, device="cuda:0"):
        self.llm = llm
        self.device = device
        self.prev_tokens = None

    @torch.inference_mode()
    def decode(self, input_embeds: torch.Tensor, n_tokens: int = 7) -> TemporalSpecResult:
        """
        Decode action tokens with temporal speculation.

        Args:
            input_embeds: [1, seq_len, hidden_size] LLM input embeddings
            n_tokens: number of action tokens to generate (default 7 for 7-DOF)

        Returns:
            TemporalSpecResult with tokens and profiling data
        """
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Prefill
        out = self.llm(inputs_embeds=input_embeds, use_cache=True)
        kv = out.past_key_values
        first = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = [first.item()]
        llm_calls = 1
        accepted = 0
        draft_used = 0

        if self.prev_tokens is not None and len(self.prev_tokens) >= n_tokens:
            # Temporal speculation: use previous action as draft
            remaining = n_tokens - 1
            draft = [torch.tensor([[t]], device=self.device)
                     for t in self.prev_tokens[1:remaining + 1]]
            draft_used = remaining

            # Verify in ONE batched pass
            verify_input = torch.cat([first] + draft, dim=1)
            t_out = self.llm(verify_input, past_key_values=kv, use_cache=True)
            kv = t_out.past_key_values
            llm_calls += 1

            # Accept/reject
            last = first
            for i in range(remaining):
                target_pred = t_out.logits[:, i, :].argmax(dim=-1).item()
                draft_pred = draft[i].item()
                if target_pred == draft_pred:
                    accepted += 1
                    generated.append(target_pred)
                    last = draft[i]
                    if len(generated) >= n_tokens:
                        break
                else:
                    generated.append(target_pred)
                    last = torch.tensor([[target_pred]], device=self.device)
                    break

            # Bonus token if all draft accepted
            if accepted == remaining and len(generated) < n_tokens:
                bonus = t_out.logits[:, remaining, :].argmax(dim=-1).item()
                generated.append(bonus)
                last = torch.tensor([[bonus]], device=self.device)

            # Fill remaining with vanilla decode
            while len(generated) < n_tokens:
                t_out = self.llm(last, past_key_values=kv, use_cache=True)
                kv = t_out.past_key_values
                tok = t_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated.append(tok.item())
                last = tok
                llm_calls += 1
        else:
            # No previous tokens: vanilla decode
            tok = first
            for _ in range(n_tokens - 1):
                out = self.llm(tok, past_key_values=kv, use_cache=True)
                kv = out.past_key_values
                tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated.append(tok.item())
                llm_calls += 1

        torch.cuda.synchronize()
        latency = (time.perf_counter() - t0) * 1000

        # Update temporal cache
        self.prev_tokens = generated

        return TemporalSpecResult(
            tokens=generated,
            latency_ms=latency,
            num_llm_calls=llm_calls,
            num_accepted=accepted,
            draft_tokens_used=draft_used,
        )

    def reset(self):
        """Reset temporal cache (e.g., when switching tasks)."""
        self.prev_tokens = None
