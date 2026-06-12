# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cosmos3-specific TeaCache hook."""

from __future__ import annotations

import torch

from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook
from vllm_omni.diffusion.cache.teacache.state import TeaCacheState

# Cosmos3's flow_shift=10 schedule clusters the early timesteps, so without a
# warmup window TeaCache caches the high-noise steps that fix the global
# composition (official TeaCache ``ret_steps`` behavior; measured on T2V at
# thresh 0.2: LPIPS 0.45 vs baseline without warmup, 0.22 with warmup 12).
COSMOS3_DEFAULT_NUM_WARMUP_STEPS = 12


class Cosmos3TeaCacheHook(TeaCacheHook):
    """TeaCacheHook that always computes the first ``num_warmup_steps`` timesteps."""

    def __init__(self, config, num_warmup_steps: int):
        super().__init__(config)
        if num_warmup_steps < 0:
            raise ValueError(f"num_warmup_steps must be >= 0, got {num_warmup_steps}")
        self.num_warmup_steps = num_warmup_steps

    def _should_compute_full_transformer(self, state: TeaCacheState, modulated_inp: torch.Tensor) -> bool:
        if state.cnt < self.num_warmup_steps:
            state.accumulated_rel_l1_distance = 0.0
            return True
        return super()._should_compute_full_transformer(state, modulated_inp)
