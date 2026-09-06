"""Supervised objectives for the specification-1.61 structured model.

Planning metrics are intentionally absent.  Candidate selection is a protected
downstream acceptance test, not a gradient owner.  Mean-state losses and
variance calibration are also separated: Gaussian NLL receives a detached
mean so a variance head cannot improve its objective by moving the physical
prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from numbers import Real

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class DynamicSetLossWeights:
    mask_dice: float = 1.0
    mask_bce: float = 1.0
    existence_focal: float = 1.0
    metric_position_huber: float = 1.0
    appearance_cosine: float = 0.25
    # Metric residuals can initially be metres while the analytic sensor prior is
    # expressed at millimetre scale.  Keep the exact detached-mean Gaussian NLL,
    # but calibrate its optimizer ownership so its variance-head gradient cannot
    # erase the perception gradients at zero-residual initialization.
    measurement_variance_nll: float = 1.0e-5
    contact_state_huber: float = 1.0
    collision_bce: float = 0.5
    process_variance_nll: float = 0.10
    known_action_trajectory: float = 1.0
    unaffected_zero_residual: float = 0.10

    def validate(self) -> DynamicSetLossWeights:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"loss weight {item.name} must be a real number")
            if not torch.isfinite(torch.tensor(float(value))) or value < 0.0:
                raise ValueError(f"loss weight {item.name} must be finite and nonnegative")
        return self


DEFAULT_DYNAMIC_SET_LOSS_WEIGHTS = DynamicSetLossWeights()

# Properties of the frozen 1.61 CPU profile, not trainable loss offsets. The
# set observation emits ``log(6.4e-5) + 20*tanh(residual)`` and dynamics clamps
# propagated fast-state log variance at -32. A diagonal Gaussian term
# ``0.5 * (log_variance + squared_error / variance)`` is therefore bounded
# below by half of the corresponding minimum log variance. The raw training
# objective remains signed; the disposable screen measures excess above this
# conservative protocol-wide lower envelope so its reduction ratio is valid.
DYNAMIC_SET_MEASUREMENT_LOG_VARIANCE_MIN = math.log(6.4e-5) - 20.0
DYNAMIC_SET_PROCESS_LOG_VARIANCE_MIN = -32.0


def dynamic_set_objective_lower_bound(
    weights: DynamicSetLossWeights = DEFAULT_DYNAMIC_SET_LOSS_WEIGHTS,
) -> float:
    """Return the conservative lower envelope of the exact signed objective."""

    weights.validate()
    return 0.5 * (
        float(weights.measurement_variance_nll) * min(0.0, DYNAMIC_SET_MEASUREMENT_LOG_VARIANCE_MIN)
        + float(weights.process_variance_nll) * min(0.0, DYNAMIC_SET_PROCESS_LOG_VARIANCE_MIN)
    )


def dynamic_set_objective_regret(
    optimization_objective: float,
    *,
    weights: DynamicSetLossWeights = DEFAULT_DYNAMIC_SET_LOSS_WEIGHTS,
) -> float:
    """Map the exact signed objective to nonnegative lower-envelope regret.

    This is an evaluation transform only. Optimizer gradients continue to use
    :func:`dynamic_set_objective` without shifting or clipping.
    """

    if (
        isinstance(optimization_objective, bool)
        or not isinstance(optimization_objective, Real)
        or not math.isfinite(float(optimization_objective))
    ):
        raise ValueError("optimization_objective must be one finite real number")
    lower_bound = dynamic_set_objective_lower_bound(weights)
    regret = float(optimization_objective) - lower_bound
    tolerance = 32.0 * math.ulp(max(1.0, abs(lower_bound)))
    if regret < -tolerance:
        raise ValueError("optimization objective lies below its protocol lower bound")
    return max(0.0, regret)


@dataclass(frozen=True)
class PerceptionLossInputs:
    mask_logits: Tensor
    target_masks: Tensor
    existence_logits: Tensor
    target_exists: Tensor
    metric_position: Tensor
    target_position: Tensor
    position_log_variance: Tensor
    appearance: Tensor
    target_appearance: Tensor


@dataclass(frozen=True)
class DynamicsLossInputs:
    predicted_state: Tensor
    target_state: Tensor
    process_log_variance: Tensor
    contact_window_mask: Tensor
    collision_logits: Tensor
    collision_target: Tensor
    collision_support: Tensor
    known_action_predictable_mask: Tensor
    relation_residual: Tensor
    unaffected_object_mask: Tensor


@dataclass(frozen=True)
class DynamicSetLosses:
    total: Tensor
    mask_dice: Tensor
    mask_bce: Tensor
    existence_focal: Tensor
    metric_position_huber: Tensor
    appearance_cosine: Tensor
    measurement_variance_nll: Tensor
    contact_state_huber: Tensor
    collision_bce: Tensor
    process_variance_nll: Tensor
    known_action_trajectory: Tensor
    unaffected_zero_residual: Tensor


def _require_float(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _require_bool(name: str, value: Tensor, shape: torch.Size | tuple[int, ...]) -> None:
    if not isinstance(value, Tensor) or value.dtype is not torch.bool:
        raise TypeError(f"{name} must be a boolean tensor")
    if value.shape != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {tuple(shape)}")


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    expanded = mask
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(value)
    numerator = torch.where(expanded, value, torch.zeros_like(value)).sum()
    denominator = expanded.sum().clamp_min(1).to(value.dtype)
    return numerator / denominator


def detached_mean_gaussian_nll(
    predicted_mean: Tensor,
    target: Tensor,
    log_variance: Tensor,
    support: Tensor,
) -> Tensor:
    """Calibrate diagonal variance without giving NLL ownership of the mean."""

    for name, value in (
        ("predicted_mean", predicted_mean),
        ("target", target),
        ("log_variance", log_variance),
    ):
        _require_float(name, value)
    if predicted_mean.shape != target.shape or log_variance.shape != target.shape:
        raise ValueError("Gaussian NLL mean, target, and log_variance shapes must match")
    _require_bool("support", support, target.shape[:-1])
    if any(
        value.device != target.device or value.dtype != target.dtype
        for value in (
            predicted_mean,
            log_variance,
        )
    ):
        raise ValueError("Gaussian NLL tensors must share device and dtype")
    residual = target - predicted_mean.detach()
    per_axis = 0.5 * (log_variance + residual.square() * torch.exp(-log_variance))
    return _masked_mean(per_axis, support)


def _focal_bce_with_logits(logits: Tensor, target: Tensor, *, gamma: float = 2.0) -> Tensor:
    probability = logits.sigmoid()
    target_float = target.to(logits.dtype)
    probability_true = torch.where(target, probability, 1.0 - probability)
    bce = F.binary_cross_entropy_with_logits(logits, target_float, reduction="none")
    return ((1.0 - probability_true).pow(gamma) * bce).mean()


def perception_losses(inputs: PerceptionLossInputs) -> dict[str, Tensor]:
    """Return the six declared perception losses for already-aligned targets."""

    for name, value in (
        ("mask_logits", inputs.mask_logits),
        ("target_masks", inputs.target_masks),
        ("existence_logits", inputs.existence_logits),
        ("metric_position", inputs.metric_position),
        ("target_position", inputs.target_position),
        ("position_log_variance", inputs.position_log_variance),
        ("appearance", inputs.appearance),
        ("target_appearance", inputs.target_appearance),
    ):
        _require_float(name, value)
    if inputs.mask_logits.shape != inputs.target_masks.shape or inputs.mask_logits.ndim != 4:
        raise ValueError("mask logits and targets must share shape [B,P+1,H,W]")
    batch, mask_classes = inputs.mask_logits.shape[:2]
    proposals = mask_classes - 1
    if proposals <= 0:
        raise ValueError("full mask logits must include background and at least one proposal")
    _require_bool("target_exists", inputs.target_exists, (batch, proposals))
    if inputs.existence_logits.shape != (batch, proposals):
        raise ValueError("existence_logits must have shape [B,P]")
    if inputs.metric_position.shape != (batch, proposals, 3):
        raise ValueError("metric_position must have shape [B,P,3]")
    if inputs.target_position.shape != inputs.metric_position.shape:
        raise ValueError("target_position must match metric_position")
    if inputs.position_log_variance.shape != inputs.metric_position.shape:
        raise ValueError("position_log_variance must match metric_position")
    if inputs.appearance.shape != inputs.target_appearance.shape:
        raise ValueError("appearance tensors must match")
    if inputs.appearance.shape[:2] != (batch, proposals):
        raise ValueError("appearance tensors must begin with [B,P]")

    if bool(torch.any((inputs.target_masks < 0.0) | (inputs.target_masks > 1.0))):
        raise ValueError("target_masks must lie in [0,1]")
    mask_maximum = inputs.mask_logits.amax(dim=1, keepdim=True)
    mask_weight = torch.exp(inputs.mask_logits - mask_maximum)
    mask_probability = mask_weight / torch.sort(mask_weight, dim=1).values.sum(
        dim=1,
        keepdim=True,
    )
    mask_support = torch.cat(
        (
            torch.ones(batch, 1, dtype=torch.bool, device=inputs.target_exists.device),
            inputs.target_exists,
        ),
        dim=1,
    )
    spatial_dims = (-2, -1)
    intersection = (mask_probability * inputs.target_masks).sum(dim=spatial_dims)
    denominator = mask_probability.sum(dim=spatial_dims) + inputs.target_masks.sum(dim=spatial_dims)
    dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    dice_loss = _masked_mean(dice, mask_support)
    mask_bce = F.binary_cross_entropy(
        mask_probability,
        inputs.target_masks,
        reduction="none",
    ).mean(dim=spatial_dims)
    mask_bce_loss = _masked_mean(mask_bce, mask_support)
    position_huber = _masked_mean(
        F.smooth_l1_loss(
            inputs.metric_position,
            inputs.target_position,
            reduction="none",
        ),
        inputs.target_exists,
    )
    # Round-off can make cosine similarity a few ulps larger than one for an
    # exactly aligned vector. A loss must remain nonnegative; this changes only
    # that impossible numerical tail and preserves the ordinary cosine
    # gradient everywhere else.
    appearance_cosine = _masked_mean(
        (
            1.0
            - F.cosine_similarity(
                inputs.appearance,
                inputs.target_appearance,
                dim=-1,
                eps=torch.finfo(inputs.appearance.dtype).eps,
            )
        ).clamp_min(0.0),
        inputs.target_exists,
    )
    return {
        "mask_dice": dice_loss,
        "mask_bce": mask_bce_loss,
        "existence_focal": _focal_bce_with_logits(
            inputs.existence_logits,
            inputs.target_exists,
        ),
        "metric_position_huber": position_huber,
        "appearance_cosine": appearance_cosine,
        "measurement_variance_nll": detached_mean_gaussian_nll(
            inputs.metric_position,
            inputs.target_position,
            inputs.position_log_variance,
            inputs.target_exists,
        ),
    }


def dynamics_losses(inputs: DynamicsLossInputs) -> dict[str, Tensor]:
    """Return declared contact/action losses with explicit causal support."""

    for name, value in (
        ("predicted_state", inputs.predicted_state),
        ("target_state", inputs.target_state),
        ("process_log_variance", inputs.process_log_variance),
        ("collision_logits", inputs.collision_logits),
        ("relation_residual", inputs.relation_residual),
    ):
        _require_float(name, value)
    if inputs.predicted_state.shape != inputs.target_state.shape:
        raise ValueError("predicted_state and target_state must match")
    if inputs.process_log_variance.shape != inputs.predicted_state.shape:
        raise ValueError("process_log_variance must match predicted_state")
    if inputs.predicted_state.ndim != 4:
        raise ValueError("state tensors must have shape [B,T,N,D]")
    batch, steps, objects = inputs.predicted_state.shape[:3]
    state_support_shape = (batch, steps, objects)
    _require_bool("contact_window_mask", inputs.contact_window_mask, state_support_shape)
    _require_bool(
        "known_action_predictable_mask",
        inputs.known_action_predictable_mask,
        state_support_shape,
    )
    if inputs.collision_logits.shape != (batch, steps, objects, objects):
        raise ValueError("collision_logits must have shape [B,T,N,N]")
    _require_bool("collision_target", inputs.collision_target, inputs.collision_logits.shape)
    _require_bool("collision_support", inputs.collision_support, inputs.collision_logits.shape)
    if inputs.relation_residual.ndim != 4 or inputs.relation_residual.shape[:3] != (
        batch,
        steps,
        objects,
    ):
        raise ValueError("relation_residual must have shape [B,T,N,Dresidual]")
    _require_bool("unaffected_object_mask", inputs.unaffected_object_mask, state_support_shape)

    state_huber = F.smooth_l1_loss(
        inputs.predicted_state,
        inputs.target_state,
        reduction="none",
    )
    collision_bce = F.binary_cross_entropy_with_logits(
        inputs.collision_logits,
        inputs.collision_target.to(inputs.collision_logits.dtype),
        reduction="none",
    )
    return {
        "contact_state_huber": _masked_mean(state_huber, inputs.contact_window_mask),
        "collision_bce": _masked_mean(collision_bce, inputs.collision_support),
        "process_variance_nll": detached_mean_gaussian_nll(
            inputs.predicted_state,
            inputs.target_state,
            inputs.process_log_variance,
            inputs.contact_window_mask | inputs.known_action_predictable_mask,
        ),
        "known_action_trajectory": _masked_mean(
            state_huber,
            inputs.known_action_predictable_mask,
        ),
        "unaffected_zero_residual": _masked_mean(
            inputs.relation_residual.square(),
            inputs.unaffected_object_mask,
        ),
    }


def dynamic_set_objective(
    perception: PerceptionLossInputs,
    dynamics: DynamicsLossInputs,
    *,
    weights: DynamicSetLossWeights = DEFAULT_DYNAMIC_SET_LOSS_WEIGHTS,
) -> DynamicSetLosses:
    """Compose the fixed physical objective; no planning loss is accepted."""

    weights.validate()
    terms = {**perception_losses(perception), **dynamics_losses(dynamics)}
    total = sum(
        (terms[item.name] * float(getattr(weights, item.name)) for item in fields(weights)),
        start=next(iter(terms.values())).new_zeros(()),
    )
    return DynamicSetLosses(total=total, **terms)


__all__ = [
    "DYNAMIC_SET_MEASUREMENT_LOG_VARIANCE_MIN",
    "DYNAMIC_SET_PROCESS_LOG_VARIANCE_MIN",
    "DynamicSetLosses",
    "DynamicSetLossWeights",
    "DEFAULT_DYNAMIC_SET_LOSS_WEIGHTS",
    "DynamicsLossInputs",
    "PerceptionLossInputs",
    "detached_mean_gaussian_nll",
    "dynamic_set_objective",
    "dynamic_set_objective_lower_bound",
    "dynamic_set_objective_regret",
    "dynamics_losses",
    "perception_losses",
]
