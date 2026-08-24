"""Runtime wrapper for alternate world-belief hypotheses."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from world_model.belief.world_belief import BeliefTrajectory, WorldBelief


@dataclass(frozen=True)
class PositionHypothesisMarginal:
    """Target-conditioned differentiable evidence for continuous modes.

    The hard object structure is shared by every mode.  Log weights remain in
    log space so an unlikely mode is retained instead of being selected or
    averaged away before the physical rollout.
    """

    negative_log_likelihood: Tensor
    posterior_log_weights: Tensor
    support: Tensor

    @property
    def posterior_weights(self) -> Tensor:
        return self.posterior_log_weights.exp()

    def supported_batch_macro_mean(self) -> Tensor:
        """Reduce this evidence once without changing its gradient graph."""

        flattened_support = self.support.flatten(1)
        per_batch_count = flattened_support.sum(dim=1)
        per_batch = self.negative_log_likelihood.flatten(1).sum(dim=1) / per_batch_count.clamp_min(
            1
        ).to(self.negative_log_likelihood.dtype)
        supported_batch = per_batch_count > 0
        return (
            per_batch * supported_batch.to(per_batch.dtype)
        ).sum() / supported_batch.sum().clamp_min(1).to(per_batch.dtype)


@dataclass
class TrajectoryHypothesisSet:
    """Equation-rollout trajectories whose continuous modes remain explicit."""

    trajectories: list[BeliefTrajectory]
    log_weights: Tensor

    def validate(self) -> TrajectoryHypothesisSet:
        if not self.trajectories:
            raise ValueError("TrajectoryHypothesisSet requires at least one trajectory")
        first = self.trajectories[0].validate()
        if self.log_weights.shape != (first.positions.shape[0], len(self.trajectories)):
            raise ValueError("trajectory hypothesis log weights must have shape [B,H]")
        if not self.log_weights.is_floating_point() or not torch.isfinite(self.log_weights).all():
            raise ValueError("trajectory hypothesis log weights must be finite floating point")
        if first.positions.device != self.log_weights.device:
            raise ValueError("trajectory hypotheses and weights must share a device")
        if first.positions.dtype != self.log_weights.dtype:
            raise ValueError("trajectory hypotheses and weights must share a dtype")
        for trajectory in self.trajectories[1:]:
            trajectory.validate()
            if trajectory.positions.shape != first.positions.shape:
                raise ValueError("all hypothesis trajectories must share position shape")
            if not torch.equal(trajectory.timestamps, first.timestamps):
                raise ValueError("all hypothesis trajectories must share timestamps")
            if not torch.equal(trajectory.active_mask, first.active_mask):
                raise ValueError("all hypothesis trajectories must share hard active structure")
            if trajectory.positions.device != self.log_weights.device:
                raise ValueError("trajectory hypotheses and weights must share a device")
            if trajectory.positions.dtype != self.log_weights.dtype:
                raise ValueError("trajectory hypotheses and weights must share a dtype")
        return self

    def marginal_position_evidence(
        self,
        target: Tensor,
        mask: Tensor,
        *,
        minimum_log_variance: float = -12.0,
        maximum_log_variance: float = 8.0,
    ) -> PositionHypothesisMarginal:
        """Evaluate the proper marginal likelihood of complete equation modes."""

        self.validate()
        reference_shape = self.trajectories[0].positions.shape
        if target.shape != reference_shape:
            raise ValueError("trajectory marginal target must have shape [B,T,N,3]")
        if mask.shape == reference_shape[:3]:
            axis_mask = mask.unsqueeze(-1).expand_as(target)
        elif mask.shape == reference_shape:
            axis_mask = mask
        else:
            raise ValueError("trajectory marginal mask must have shape [B,T,N] or [B,T,N,3]")
        if axis_mask.dtype is not torch.bool:
            raise TypeError("trajectory marginal mask must be boolean")
        if target.device != self.log_weights.device or target.dtype != self.log_weights.dtype:
            raise ValueError("trajectory target must share hypothesis device and dtype")
        if not torch.isfinite(target).all():
            raise ValueError("trajectory marginal target must be finite")
        if not math.isfinite(minimum_log_variance) or not math.isfinite(maximum_log_variance):
            raise ValueError("trajectory marginal variance bounds must be finite")
        if minimum_log_variance > maximum_log_variance:
            raise ValueError("trajectory marginal variance bounds are reversed")

        means = torch.stack([item.positions for item in self.trajectories], dim=3)
        log_variances = torch.stack(
            [
                item.fast_log_variance[..., :3].clamp(
                    minimum_log_variance,
                    maximum_log_variance,
                )
                for item in self.trajectories
            ],
            dim=3,
        )
        residual = target.unsqueeze(3) - means
        coordinate_log_probability = -0.5 * (
            math.log(2.0 * math.pi) + log_variances + residual.square() * torch.exp(-log_variances)
        )
        component_log_likelihood = (coordinate_log_probability * axis_mask.unsqueeze(3)).sum(dim=-1)
        normalized_prior = torch.log_softmax(self.log_weights, dim=-1)
        joint = normalized_prior[:, None, None, :] + component_log_likelihood
        marginal_log_likelihood = torch.logsumexp(joint, dim=-1)
        posterior_log_weights = joint - marginal_log_likelihood.unsqueeze(-1)
        support = axis_mask.any(dim=-1)
        posterior_log_weights = torch.where(
            support.unsqueeze(-1),
            posterior_log_weights,
            normalized_prior[:, None, None, :].expand_as(posterior_log_weights),
        )
        negative_log_likelihood = torch.where(
            support,
            -marginal_log_likelihood,
            marginal_log_likelihood * 0.0,
        )
        return PositionHypothesisMarginal(
            negative_log_likelihood=negative_log_likelihood,
            posterior_log_weights=posterior_log_weights,
            support=support,
        )

    def marginal_position_nll(
        self,
        target: Tensor,
        mask: Tensor,
        *,
        minimum_log_variance: float = -12.0,
        maximum_log_variance: float = 8.0,
    ) -> Tensor:
        """Return a supported-batch macro mean trajectory mixture NLL."""

        result = self.marginal_position_evidence(
            target,
            mask,
            minimum_log_variance=minimum_log_variance,
            maximum_log_variance=maximum_log_variance,
        )
        return result.supported_batch_macro_mean()


@dataclass
class HypothesisSet:
    """A small list of beliefs with batch-wise log weights ``[B,H]``."""

    beliefs: list[WorldBelief]
    log_weights: Tensor

    def validate(self) -> HypothesisSet:
        if not self.beliefs:
            raise ValueError("HypothesisSet requires at least one belief")
        batch = self.beliefs[0].batch_size
        if self.log_weights.shape != (batch, len(self.beliefs)):
            raise ValueError("log_weights must have shape [B,H]")
        if not torch.isfinite(self.log_weights).all():
            raise ValueError("hypothesis log weights must be finite")
        if not self.log_weights.is_floating_point():
            raise TypeError("hypothesis log weights must be floating point")
        for belief in self.beliefs:
            belief.validate()
            if belief.batch_size != batch:
                raise ValueError("all hypotheses must have the same batch size")
            if belief.device != self.log_weights.device:
                raise ValueError("hypothesis weights and beliefs must share a device")
            if belief.dtype != self.log_weights.dtype:
                raise ValueError("hypothesis weights and beliefs must share a dtype")
        return self

    @classmethod
    def singleton(cls, belief: WorldBelief) -> HypothesisSet:
        return cls(
            beliefs=[belief],
            log_weights=torch.zeros(
                belief.batch_size,
                1,
                device=belief.device,
                dtype=belief.dtype,
            ),
        )

    @property
    def normalized_weights(self) -> Tensor:
        return torch.softmax(self.log_weights, dim=-1)

    def reweight(self, log_likelihood: Tensor) -> HypothesisSet:
        if log_likelihood.shape != self.log_weights.shape:
            raise ValueError("log_likelihood must have shape [B,H]")
        updated = self.log_weights + log_likelihood
        updated = updated - torch.logsumexp(updated, dim=-1, keepdim=True)
        return HypothesisSet(self.beliefs.copy(), updated)

    def _validate_shared_object_structure(self) -> None:
        reference = self.beliefs[0].objects
        for belief in self.beliefs[1:]:
            if not torch.equal(belief.objects.object_id, reference.object_id):
                raise ValueError("continuous hypotheses must share object identity")
            if not torch.equal(belief.objects.active, reference.active):
                raise ValueError("continuous hypotheses must share active object structure")

    def marginal_position_evidence(
        self,
        target: Tensor,
        mask: Tensor,
        *,
        minimum_log_variance: float = -12.0,
        maximum_log_variance: float = 8.0,
    ) -> PositionHypothesisMarginal:
        """Evaluate a Gaussian mixture without collapsing continuous modes.

        ``target`` is ``[B,N,3]``. ``mask`` may be object-wise ``[B,N]`` or
        coordinate-wise ``[B,N,3]``.  Each hypothesis contributes the
        position mean and the canonical first three fast-state log variances;
        shared hard identity/lifecycle tensors are never relaxed.
        """

        self.validate()
        self._validate_shared_object_structure()
        reference_shape = self.beliefs[0].objects.position.shape
        if target.shape != reference_shape:
            raise ValueError("position marginal target must have shape [B,N,3]")
        if mask.shape == reference_shape[:2]:
            axis_mask = mask.unsqueeze(-1).expand_as(target)
        elif mask.shape == reference_shape:
            axis_mask = mask
        else:
            raise ValueError("position marginal mask must have shape [B,N] or [B,N,3]")
        if axis_mask.dtype is not torch.bool:
            raise TypeError("position marginal mask must be boolean")
        if target.device != self.log_weights.device or target.dtype != self.log_weights.dtype:
            raise ValueError("position marginal target must share hypothesis device and dtype")
        if not torch.isfinite(target).all():
            raise ValueError("position marginal target must be finite")
        if not math.isfinite(minimum_log_variance) or not math.isfinite(maximum_log_variance):
            raise ValueError("position marginal variance bounds must be finite")
        if minimum_log_variance > maximum_log_variance:
            raise ValueError("position marginal variance bounds are reversed")

        means = torch.stack(
            [belief.objects.position for belief in self.beliefs],
            dim=2,
        )
        log_variances = torch.stack(
            [
                belief.objects.fast_log_variance[..., :3].clamp(
                    minimum_log_variance,
                    maximum_log_variance,
                )
                for belief in self.beliefs
            ],
            dim=2,
        )
        residual = target.unsqueeze(2) - means
        coordinate_log_probability = -0.5 * (
            math.log(2.0 * math.pi) + log_variances + residual.square() * torch.exp(-log_variances)
        )
        component_log_likelihood = (coordinate_log_probability * axis_mask.unsqueeze(2)).sum(dim=-1)
        normalized_prior = torch.log_softmax(self.log_weights, dim=-1)
        joint = normalized_prior.unsqueeze(1) + component_log_likelihood
        marginal_log_likelihood = torch.logsumexp(joint, dim=-1)
        posterior_log_weights = joint - marginal_log_likelihood.unsqueeze(-1)
        support = axis_mask.any(dim=-1)
        posterior_log_weights = torch.where(
            support.unsqueeze(-1),
            posterior_log_weights,
            normalized_prior.unsqueeze(1).expand_as(posterior_log_weights),
        )
        negative_log_likelihood = torch.where(
            support,
            -marginal_log_likelihood,
            marginal_log_likelihood * 0.0,
        )
        return PositionHypothesisMarginal(
            negative_log_likelihood=negative_log_likelihood,
            posterior_log_weights=posterior_log_weights,
            support=support,
        )

    def marginal_position_nll(
        self,
        target: Tensor,
        mask: Tensor,
        *,
        minimum_log_variance: float = -12.0,
        maximum_log_variance: float = 8.0,
    ) -> Tensor:
        """Return a supported-batch macro mean of mixture position NLL."""

        result = self.marginal_position_evidence(
            target,
            mask,
            minimum_log_variance=minimum_log_variance,
            maximum_log_variance=maximum_log_variance,
        )
        return result.supported_batch_macro_mean()

    def posterior_expected_position(self, evidence: PositionHypothesisMarginal) -> Tensor:
        """Moment-read the target-conditioned posterior without mutating modes."""

        expected_shape = (
            self.beliefs[0].batch_size,
            self.beliefs[0].objects.max_objects,
            len(self.beliefs),
        )
        if evidence.posterior_log_weights.shape != expected_shape:
            raise ValueError("posterior hypothesis weights have incompatible shape")
        means = torch.stack(
            [belief.objects.position for belief in self.beliefs],
            dim=2,
        )
        return torch.sum(evidence.posterior_weights.unsqueeze(-1) * means, dim=2)

    def map(self, function: Callable[[WorldBelief], WorldBelief]) -> HypothesisSet:
        return HypothesisSet([function(item) for item in self.beliefs], self.log_weights)

    def rollout(
        self,
        function: Callable[[WorldBelief, Tensor | Sequence[float]], BeliefTrajectory],
        query_times: Tensor | Sequence[float],
    ) -> TrajectoryHypothesisSet:
        """Propagate every continuous mode through the same dynamics callable."""

        self.validate()
        return TrajectoryHypothesisSet(
            [function(item, query_times) for item in self.beliefs],
            self.log_weights,
        ).validate()

    def clone(self) -> HypothesisSet:
        return HypothesisSet(
            [belief.clone() for belief in self.beliefs],
            self.log_weights.clone(),
        )

    def detach(self) -> HypothesisSet:
        return HypothesisSet(
            [belief.detach() for belief in self.beliefs],
            self.log_weights.detach(),
        )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> HypothesisSet:
        return HypothesisSet(
            [belief.to(device=device, dtype=dtype) for belief in self.beliefs],
            self.log_weights.to(device=device, dtype=dtype),
        )

    def best(self, batch_index: int = 0) -> WorldBelief:
        """Return the highest-weight hypothesis for one batch element.

        Hypotheses contain whole batched beliefs, so this chooses one common
        hypothesis object.  Per-example branching can be added without changing
        the wrapper contract.
        """

        index = int(self.log_weights[batch_index].argmax().item())
        return self.beliefs[index]
