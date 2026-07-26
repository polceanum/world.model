"""Bounded learned residual correction layered on the analytic proposal."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class LearnedCorrection:
    mean_delta: Tensor
    log_variance_delta: Tensor
    state_gate: Tensor
    mode_logit_delta: Tensor
    existence_delta: Tensor
    visibility_delta: Tensor


class LearnedFastCorrector(nn.Module):
    """Small per-pair corrector; final residual heads start at zero."""

    def __init__(
        self,
        *,
        fast_state_dim: int,
        num_motion_modes: int,
        hidden_dim: int = 128,
        modality_count: int = 8,
        modality_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        self.fast_state_dim = fast_state_dim
        self.num_motion_modes = num_motion_modes
        self.modality_embedding = nn.Embedding(modality_count, modality_embedding_dim)
        # Innovation is summarized by mean, mean absolute, max absolute, norm,
        # association cost, ambiguity, visibility, and elapsed time.
        input_dim = fast_state_dim * 2 + 8 + num_motion_modes + modality_embedding_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, fast_state_dim)
        self.variance_head = nn.Linear(hidden_dim, fast_state_dim)
        self.gate_head = nn.Linear(hidden_dim, fast_state_dim)
        self.mode_head = nn.Linear(hidden_dim, num_motion_modes)
        self.existence_head = nn.Linear(hidden_dim, 1)
        self.visibility_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.variance_head.weight)
        nn.init.zeros_(self.variance_head.bias)
        nn.init.constant_(self.gate_head.bias, -2.0)
        nn.init.zeros_(self.mode_head.weight)
        nn.init.zeros_(self.mode_head.bias)
        nn.init.zeros_(self.existence_head.weight)
        nn.init.zeros_(self.existence_head.bias)
        nn.init.zeros_(self.visibility_head.weight)
        nn.init.zeros_(self.visibility_head.bias)

    @staticmethod
    def _innovation_summary(
        whitened_innovation: Tensor,
        association_cost: Tensor,
        ambiguity: Tensor,
        visibility: Tensor,
        elapsed_time: Tensor,
    ) -> Tensor:
        return torch.cat(
            (
                whitened_innovation.mean(dim=-1, keepdim=True),
                whitened_innovation.abs().mean(dim=-1, keepdim=True),
                whitened_innovation.abs().amax(dim=-1, keepdim=True),
                torch.linalg.vector_norm(whitened_innovation, dim=-1, keepdim=True),
                association_cost.unsqueeze(-1).nan_to_num(posinf=100.0),
                ambiguity.to(whitened_innovation.dtype).unsqueeze(-1),
                visibility.unsqueeze(-1),
                elapsed_time.unsqueeze(-1),
            ),
            dim=-1,
        )

    def forward(
        self,
        *,
        prior_fast_state: Tensor,
        prior_log_variance: Tensor,
        whitened_innovation: Tensor,
        association_cost: Tensor,
        ambiguity: Tensor,
        visibility: Tensor,
        elapsed_time: Tensor,
        motion_mode_logits: Tensor,
        modality_index: Tensor,
    ) -> LearnedCorrection:
        if prior_fast_state.shape[-1] != self.fast_state_dim:
            raise ValueError("learned corrector fast-state dimension mismatch")
        summary = self._innovation_summary(
            whitened_innovation,
            association_cost,
            ambiguity,
            visibility,
            elapsed_time,
        )
        mode_probability = motion_mode_logits.softmax(dim=-1)
        modality = self.modality_embedding(
            modality_index.clamp(0, self.modality_embedding.num_embeddings - 1)
        )
        hidden = self.network(
            torch.cat(
                (
                    prior_fast_state,
                    prior_log_variance,
                    summary,
                    mode_probability,
                    modality,
                ),
                dim=-1,
            )
        )
        return LearnedCorrection(
            mean_delta=torch.tanh(self.mean_head(hidden)),
            log_variance_delta=0.5 * torch.tanh(self.variance_head(hidden)),
            state_gate=torch.sigmoid(self.gate_head(hidden)),
            mode_logit_delta=self.mode_head(hidden),
            existence_delta=self.existence_head(hidden).squeeze(-1),
            visibility_delta=self.visibility_head(hidden).squeeze(-1),
        )
