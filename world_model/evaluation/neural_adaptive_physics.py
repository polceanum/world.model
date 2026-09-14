"""Train and evaluate a compact neural physical-belief adapter.

This is a development bridge, not a protected promotion.  It proves that a
single learned model can acquire reusable weights across episodes and adapt a
new object's interpretable physical state from public evidence at runtime.
The qualified analytic geometry, rigid contact solver, and planner remain in
place, providing a stable baseline and an exact rollback path.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from world_model.belief import RigidPrimitive, WorldBelief, slow_packing_map
from world_model.evaluation import adaptive_physics_scale as analytic_gate
from world_model.evaluation import multicontact_six_dof as state_gate
from world_model.evaluation import visual_dynamic_scale as visual_gate
from world_model.evaluation.capability_summary import (
    CAPABILITY_SUMMARY_SCHEMA,
    CapabilityRunSummary,
    write_capability_summary,
)
from world_model.identification import (
    EvidenceTransformerConfig,
    NeuralPhysicsAdapter,
    PhysicsEvidenceKind,
    event_transition_token,
    free_motion_tokens,
    normalized_parameter_targets,
)
from world_model.simulator import RigidBodyState, advance_rigid_bodies_6dof
from world_model.utils.io import atomic_write_text
from world_model.utils.run_artifacts import inventory_runs, write_run_manifest
from world_model.visualisation.progress import build_progress_dashboard, write_run_report

NEURAL_ADAPTIVE_PHYSICS_SCHEMA = "world_model_neural_adaptive_physics_v1"
DEFAULT_TRAINING_STEPS = 2_048
DEFAULT_BATCH_SIZE = 128
DEFAULT_TRAINING_SEED = 91_700
DEFAULT_DEVELOPMENT_SEED = 92_700
DEFAULT_OOD_SEED = 93_700
_PARAMETER_NAMES = ("mass", "drag", "restitution", "friction")


@dataclass(frozen=True, slots=True)
class SyntheticPhysicsBatch:
    """Public evidence paired with private training-only parameter targets."""

    evidence: Tensor
    valid: Tensor
    target_belief_values: Tensor


@dataclass(frozen=True, slots=True)
class NeuralPhysicsMetrics:
    example_count: int
    mean_relative_error: float
    p95_relative_error: float
    maximum_relative_error: float
    by_parameter: dict[str, float]
    prefix_mean_relative_error: dict[str, float]
    permutation_maximum_difference: float


@dataclass(frozen=True, slots=True)
class NeuralAdaptivePhysicsResult:
    schema: str
    protocol_sha256: str
    training_steps: int
    training_batch_size: int
    training_examples: int
    training_seconds: float
    training_loss_initial: float
    training_loss_final: float
    training_curve: tuple[dict[str, float], ...]
    development: NeuralPhysicsMetrics
    compositional_ood: NeuralPhysicsMetrics
    scenarios: tuple[analytic_gate.AdaptivePhysicsScenarioResult, ...]
    planning: tuple[state_gate.MultiContactPlanningResult, ...]
    learned_parameter_count: int
    nonzero_learned_parameter_count: int
    changed_parameter_count: int
    learned_weight_bytes: int
    checkpoint_sha256: str
    gate_failures: tuple[str, ...]
    passed: bool
    evaluation_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def neural_adaptive_protocol_sha256(*, training_steps: int, batch_size: int) -> str:
    """Bind the architecture, deterministic streams, and public evaluation ladder."""

    payload = {
        "schema": NEURAL_ADAPTIVE_PHYSICS_SCHEMA,
        "architecture": asdict(EvidenceTransformerConfig()),
        "training": {
            "steps": training_steps,
            "batch_size": batch_size,
            "seed": DEFAULT_TRAINING_SEED,
            "optimizer": "AdamW",
            "learning_rate": 7.5e-4,
            "weight_decay": 1.0e-4,
        },
        "development": {"seed": DEFAULT_DEVELOPMENT_SEED, "examples": 2_048},
        "compositional_ood": {"seed": DEFAULT_OOD_SEED, "examples": 2_048},
        "public_counts": [4, 6, 8],
        "planning_candidates": [8, 32],
        "planning_used_as_training_loss": False,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _uniform(
    generator: torch.Generator,
    shape: tuple[int, ...],
    lower: float,
    upper: float,
) -> Tensor:
    return lower + (upper - lower) * torch.rand(shape, generator=generator)


def _unit_vectors(generator: torch.Generator, batch_size: int) -> Tensor:
    value = torch.randn((batch_size, 3), generator=generator)
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1.0e-6)


def _positions_around_event(velocity_at_event: Tensor, drag: Tensor, times: Tensor) -> Tensor:
    expanded_time = times.unsqueeze(0)
    scale = -torch.expm1(-drag.unsqueeze(-1) * expanded_time) / drag.unsqueeze(-1)
    return scale.unsqueeze(-1) * velocity_at_event.unsqueeze(1)


def _measurement_noise(
    generator: torch.Generator,
    reference: Tensor,
    *,
    ood: bool,
) -> Tensor:
    maximum = 0.0035 if ood else 0.0020
    scale = _uniform(generator, (reference.shape[0], 1, 1), 0.0002, maximum)
    return torch.randn(reference.shape, generator=generator) * scale


def synthetic_physics_batch(
    batch_size: int,
    *,
    generator: torch.Generator,
    config: EvidenceTransformerConfig | None = None,
    ood: bool = False,
) -> SyntheticPhysicsBatch:
    """Generate compact causal evidence without placing truth in model inputs."""

    if batch_size <= 0:
        raise ValueError("synthetic physics batch size must be positive")
    adapter_config = (config or EvidenceTransformerConfig()).validate()
    if ood:
        # Compositional edge strata remain inside declared physical bounds but
        # outside the central ranges used by optimizer draws.
        selector = torch.randint(0, 2, (batch_size, 4), generator=generator)

        def edge_sample(index: int, bounds: tuple[float, float], inset: float) -> Tensor:
            lower, upper = bounds
            span = upper - lower
            low = _uniform(generator, (batch_size,), lower, lower + inset * span)
            high = _uniform(generator, (batch_size,), upper - inset * span, upper)
            return torch.where(selector[:, index].bool(), high, low)

        mass = edge_sample(0, adapter_config.mass_bounds, 0.18)
        drag = edge_sample(1, adapter_config.drag_bounds, 0.18)
        restitution = edge_sample(2, adapter_config.restitution_bounds, 0.18)
        friction = edge_sample(3, adapter_config.friction_bounds, 0.18)
    else:
        mass = _uniform(generator, (batch_size,), 0.60, 2.00)
        drag = _uniform(generator, (batch_size,), 0.05, 0.30)
        restitution = _uniform(generator, (batch_size,), 0.28, 0.92)
        friction = _uniform(generator, (batch_size,), 0.05, 0.65)

    free_times = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
    free_direction = _unit_vectors(generator, batch_size)
    free_speed = _uniform(generator, (batch_size, 1), 0.7, 2.0)
    free_velocity = free_direction * free_speed
    free_scale = -torch.expm1(-drag[:, None] * free_times[None]) / drag[:, None]
    free_positions = free_scale.unsqueeze(-1) * free_velocity.unsqueeze(1)
    free_positions = free_positions + _measurement_noise(
        generator,
        free_positions,
        ood=ood,
    )
    free = free_motion_tokens(
        free_positions,
        free_times.unsqueeze(0).expand(batch_size, -1),
    )

    impulse_before_times = torch.tensor([-1.0, -0.5, 0.0])
    contact_before_times = torch.tensor([-0.30, -0.15, 0.0])
    after_times = torch.tensor([0.0, 0.15, 0.30])
    pre_impulse_velocity = _unit_vectors(generator, batch_size) * _uniform(
        generator,
        (batch_size, 1),
        0.5,
        1.8,
    )
    impulse = _unit_vectors(generator, batch_size) * _uniform(
        generator,
        (batch_size, 1),
        0.35,
        1.20,
    )
    post_impulse_velocity = pre_impulse_velocity + impulse / mass.unsqueeze(-1)
    impulse_before = _positions_around_event(
        pre_impulse_velocity,
        drag,
        impulse_before_times,
    )
    impulse_after = _positions_around_event(
        post_impulse_velocity,
        drag,
        after_times,
    )
    impulse_before += _measurement_noise(generator, impulse_before, ood=ood)
    impulse_after += _measurement_noise(generator, impulse_after, ood=ood)
    impulse_token = event_transition_token(
        PhysicsEvidenceKind.KNOWN_IMPULSE,
        impulse_before,
        impulse_before_times.unsqueeze(0).expand(batch_size, -1),
        impulse_after,
        after_times.unsqueeze(0).expand(batch_size, -1),
        known_impulse_world=impulse,
    )

    normal = _unit_vectors(generator, batch_size)
    tangent_seed = _unit_vectors(generator, batch_size)
    tangent = tangent_seed - (tangent_seed * normal).sum(dim=-1, keepdim=True) * normal
    tangent = tangent / torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(1.0e-6)
    incoming_speed = _uniform(generator, (batch_size, 1), 1.0, 4.5)
    tangent_speed = _uniform(generator, (batch_size, 1), 1.0, 4.5)
    collision_before_velocity = -incoming_speed * normal + tangent_speed * tangent
    maximum_tangent_delta = (
        friction.unsqueeze(-1) * (1.0 + restitution.unsqueeze(-1)) * incoming_speed
    )
    retained_tangent = (tangent_speed - maximum_tangent_delta).clamp_min(0.0) * tangent
    collision_after_velocity = (
        restitution.unsqueeze(-1) * incoming_speed * normal + retained_tangent
    )
    collision_before = _positions_around_event(
        collision_before_velocity,
        drag,
        contact_before_times,
    )
    collision_after = _positions_around_event(
        collision_after_velocity,
        drag,
        after_times,
    )
    collision_before += _measurement_noise(generator, collision_before, ood=ood)
    collision_after += _measurement_noise(generator, collision_after, ood=ood)
    collision_token = event_transition_token(
        PhysicsEvidenceKind.BOUNDARY_CONTACT,
        collision_before,
        contact_before_times.unsqueeze(0).expand(batch_size, -1),
        collision_after,
        after_times.unsqueeze(0).expand(batch_size, -1),
        contact_normal_world=normal,
    )

    evidence = torch.cat((free, impulse_token, collision_token), dim=1)
    valid = torch.ones(evidence.shape[:2], dtype=torch.bool)
    target = torch.stack(
        (
            mass.log(),
            drag.log(),
            torch.logit(restitution),
            torch.logit(friction),
        ),
        dim=-1,
    )
    return SyntheticPhysicsBatch(evidence, valid, target)


def _permuted(evidence: Tensor, valid: Tensor, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    order = torch.randperm(evidence.shape[1], generator=generator)
    return evidence[:, order], valid[:, order]


def train_neural_physics_adapter(
    *,
    steps: int = DEFAULT_TRAINING_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_TRAINING_SEED,
) -> tuple[NeuralPhysicsAdapter, tuple[dict[str, float], ...], float, int]:
    """Train the compact adapter with deterministic streamed synthetic draws."""

    if steps <= 0 or batch_size <= 0:
        raise ValueError("training steps and batch size must be positive")
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    model = NeuralPhysicsAdapter().train()
    initial_state = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=7.5e-4, weight_decay=1.0e-4)
    trace: list[dict[str, float]] = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        batch = synthetic_physics_batch(
            batch_size,
            generator=generator,
            config=model.config,
        )
        target = normalized_parameter_targets(batch.target_belief_values, model.config)
        free_evidence, free_valid = _permuted(
            batch.evidence[:, :3],
            batch.valid[:, :3],
            generator,
        )
        action_evidence, action_valid = _permuted(
            batch.evidence[:, :4],
            batch.valid[:, :4],
            generator,
        )
        full_evidence, full_valid = _permuted(batch.evidence, batch.valid, generator)
        free_prediction = model(free_evidence, free_valid)
        action_prediction = model(action_evidence, action_valid)
        full_prediction = model(full_evidence, full_valid)
        free_loss = F.smooth_l1_loss(
            free_prediction.normalized_mean[:, 1],
            target[:, 1],
            beta=0.02,
        )
        action_loss = F.smooth_l1_loss(
            action_prediction.normalized_mean[:, :2],
            target[:, :2],
            beta=0.02,
        )
        full_loss = F.smooth_l1_loss(
            full_prediction.normalized_mean,
            target,
            beta=0.02,
        )
        detached_residual = (full_prediction.normalized_mean.detach() - target).square()
        uncertainty_loss = (
            detached_residual * torch.exp(-full_prediction.normalized_log_variance)
            + full_prediction.normalized_log_variance
        ).mean()
        loss = 0.20 * free_loss + 0.30 * action_loss + full_loss + 0.01 * uncertainty_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 16 == 0 or step == steps:
            trace.append(
                {
                    "step": float(step),
                    "loss": float(full_loss.detach()),
                    "optimized_loss": float(loss.detach()),
                    "full_parameter_loss": float(full_loss.detach()),
                }
            )
    changed_parameters = sum(
        int(torch.count_nonzero(parameter.detach() != initial_state[name]))
        for name, parameter in model.named_parameters()
    )
    return model.eval(), tuple(trace), time.perf_counter() - started, changed_parameters


def _physical_values(belief_values: Tensor) -> Tensor:
    return torch.stack(
        (
            belief_values[..., 0].exp(),
            belief_values[..., 1].exp(),
            belief_values[..., 2].sigmoid(),
            belief_values[..., 3].sigmoid(),
        ),
        dim=-1,
    )


def evaluate_neural_physics_adapter(
    model: NeuralPhysicsAdapter,
    *,
    seed: int,
    example_count: int = 2_048,
    ood: bool = False,
) -> NeuralPhysicsMetrics:
    generator = torch.Generator().manual_seed(seed)
    batch = synthetic_physics_batch(
        example_count,
        generator=generator,
        config=model.config,
        ood=ood,
    )
    target = _physical_values(batch.target_belief_values)
    prefix_errors: dict[str, float] = {}
    predictions = {}
    with torch.no_grad():
        for name, count in (("free_motion", 3), ("known_action", 4), ("contact", 5)):
            prediction = model(batch.evidence[:, :count], batch.valid[:, :count])
            physical = _physical_values(prediction.belief_values)
            predictions[name] = prediction
            prefix_errors[name] = float(((physical - target).abs() / target).mean())
        permuted_evidence, permuted_valid = _permuted(batch.evidence, batch.valid, generator)
        permuted_prediction = model(permuted_evidence, permuted_valid)
    final_physical = _physical_values(predictions["contact"].belief_values)
    relative = (final_physical - target).abs() / target
    permutation_difference = float(
        (predictions["contact"].normalized_mean - permuted_prediction.normalized_mean).abs().max()
    )
    return NeuralPhysicsMetrics(
        example_count=example_count,
        mean_relative_error=float(relative.mean()),
        p95_relative_error=float(torch.quantile(relative, 0.95)),
        maximum_relative_error=float(relative.max()),
        by_parameter={
            name: float(relative[:, index].mean()) for index, name in enumerate(_PARAMETER_NAMES)
        },
        prefix_mean_relative_error=prefix_errors,
        permutation_maximum_difference=permutation_difference,
    )


def _public_neural_calibrator(
    model: NeuralPhysicsAdapter,
    belief: WorldBelief,
    truth: RigidBodyState,
    truth_by_slot: Tensor,
) -> tuple[WorldBelief, int, int, float, tuple[dict[str, Any], ...]]:
    """Adapt one public RGB-D scene without exposing private values to the model."""

    adapter = copy.deepcopy(model).to(dtype=belief.dtype).eval()
    initial = belief.objects.slow_log_variance.clone()
    source = belief
    stages = [analytic_gate._parameter_stage("neutral priors", belief, truth, truth_by_slot)]
    evidence_rows = []
    maximum_fit_error = 0.0
    free_times = belief.timestamp.new_tensor([0.0, 0.5, 1.0, 1.5, 2.0])
    before_times = belief.timestamp.new_tensor([-0.30, -0.15, 0.0])
    after_times = belief.timestamp.new_tensor([0.0, 0.15, 0.30])
    normal = belief.timestamp.new_tensor([-1.0, 0.0, 0.0])
    for slot in range(belief.objects.max_objects):
        truth_index = int(truth_by_slot[slot])
        direction = -1.0 if slot % 2 else 1.0
        state, free_positions, fit_error = analytic_gate._motion_trace(
            truth,
            truth_index,
            free_times,
            belief.timestamp.new_tensor([0.0, 0.0, 4.0]),
            belief.timestamp.new_tensor([0.9 + 0.04 * slot, direction * 0.28, 0.08]),
        )
        maximum_fit_error = max(maximum_fit_error, fit_error)
        free = free_motion_tokens(free_positions.unsqueeze(0), free_times.unsqueeze(0))[0]

        impulse = belief.timestamp.new_tensor([0.96 + 0.05 * slot, direction * 0.34, 0.12])
        external = torch.zeros_like(state.velocity)
        external[truth_index] = impulse
        impulse_after_positions = [free_positions[-1]]
        for step, endpoint in enumerate((2.15, 2.30)):
            state, _ = advance_rigid_bodies_6dof(
                state,
                0.15,
                analytic_gate._PROBE_PHYSICS,
                external_impulse=external if step == 0 else None,
            )
            observed, fit_error = analytic_gate._observe_single_position(state, endpoint)
            impulse_after_positions.append(observed)
            maximum_fit_error = max(maximum_fit_error, fit_error)
        impulse_token = event_transition_token(
            PhysicsEvidenceKind.KNOWN_IMPULSE,
            free_positions[-3:].unsqueeze(0),
            free_times[-3:].unsqueeze(0),
            torch.stack(impulse_after_positions).unsqueeze(0),
            belief.timestamp.new_tensor([[2.0, 2.15, 2.30]]),
            known_impulse_world=impulse.unsqueeze(0),
        )[0]

        restitution = truth.restitution[truth_index, 0]
        friction = truth.friction[truth_index, 0]
        before_velocity = belief.timestamp.new_tensor([4.0, 5.0, 0.0])
        after_velocity = torch.stack(
            (
                -4.0 * restitution,
                5.0 - 4.0 * friction * (1.0 + restitution),
                restitution * 0.0,
            )
        )
        support = (
            truth.half_extents[truth_index, 0]
            if int(truth.primitive[truth_index]) == int(RigidPrimitive.BOX)
            else truth.radius[truth_index, 0]
        )
        collision_position = torch.stack(
            (support.new_tensor(1.75) - support, support * 0.0, support.new_tensor(4.0))
        )
        collision_before, before_error = analytic_gate._analytic_probe_positions(
            truth,
            truth_index,
            before_times,
            collision_position,
            before_velocity,
        )
        collision_after, after_error = analytic_gate._analytic_probe_positions(
            truth,
            truth_index,
            after_times,
            collision_position,
            after_velocity,
        )
        maximum_fit_error = max(maximum_fit_error, before_error, after_error)
        collision_token = event_transition_token(
            PhysicsEvidenceKind.BOUNDARY_CONTACT,
            collision_before.unsqueeze(0),
            before_times.unsqueeze(0),
            collision_after.unsqueeze(0),
            after_times.unsqueeze(0),
            contact_normal_world=normal.unsqueeze(0),
        )[0]
        evidence_rows.append(torch.cat((free, impulse_token, collision_token), dim=0))

    evidence = torch.stack(evidence_rows).unsqueeze(0)
    valid = torch.ones(evidence.shape[:-1], dtype=torch.bool, device=evidence.device)
    with torch.no_grad():
        free_belief = adapter.adapt_belief(source, evidence[..., :3, :], valid[..., :3])
        action_belief = adapter.adapt_belief(source, evidence[..., :4, :], valid[..., :4])
        adapted = adapter.adapt_belief(source, evidence, valid)
    stages.append(
        analytic_gate._parameter_stage(
            "neural public free-motion evidence",
            free_belief,
            truth,
            truth_by_slot,
        )
    )
    stages.append(
        analytic_gate._parameter_stage(
            "neural public known-action evidence",
            action_belief,
            truth,
            truth_by_slot,
        )
    )
    stages.append(
        analytic_gate._parameter_stage(
            "neural public contact evidence",
            adapted,
            truth,
            truth_by_slot,
        )
    )
    packing = slow_packing_map(adapted.objects)
    indices = torch.cat(
        [
            torch.arange(packing[name].start, packing[name].stop)
            for name in ("log_mass", "log_drag", "restitution_logit", "friction_logit")
        ]
    )
    contracted = (adapted.objects.slow_log_variance[..., indices] < initial[..., indices]).all(
        dim=-1
    )
    contracted_count = int(contracted[adapted.objects.active].sum())
    return (
        adapted,
        4 * adapted.objects.max_objects,
        contracted_count,
        maximum_fit_error,
        tuple(stages),
    )


def _checkpoint_payload(
    model: NeuralPhysicsAdapter,
    *,
    result: NeuralAdaptivePhysicsResult,
) -> dict[str, Any]:
    return {
        "schema": NEURAL_ADAPTIVE_PHYSICS_SCHEMA,
        "model": "NeuralPhysicsAdapter",
        "config": asdict(model.config),
        "protocol_sha256": result.protocol_sha256,
        "training_steps": result.training_steps,
        "training_batch_size": result.training_batch_size,
        "training_examples": result.training_examples,
        "learned_parameter_count": result.learned_parameter_count,
        "nonzero_learned_parameter_count": result.nonzero_learned_parameter_count,
        "changed_parameter_count": result.changed_parameter_count,
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_neural_physics_adapter(
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> NeuralPhysicsAdapter:
    """Load and strictly validate a published weights-only adapter checkpoint."""

    payload = torch.load(
        Path(checkpoint).expanduser(),
        map_location=map_location,
        weights_only=True,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("neural adapter checkpoint must contain a mapping")
    if payload.get("schema") != NEURAL_ADAPTIVE_PHYSICS_SCHEMA:
        raise ValueError("neural adapter checkpoint schema does not match")
    if payload.get("model") != "NeuralPhysicsAdapter":
        raise ValueError("neural adapter checkpoint model does not match")
    raw_config = payload.get("config")
    state_dict = payload.get("state_dict")
    if not isinstance(raw_config, Mapping) or not isinstance(state_dict, Mapping):
        raise ValueError("neural adapter checkpoint is missing config or state_dict")
    try:
        config = EvidenceTransformerConfig(**dict(raw_config)).validate()
    except (TypeError, ValueError) as error:
        raise ValueError("neural adapter checkpoint config is invalid") from error
    if not state_dict or not all(isinstance(value, Tensor) for value in state_dict.values()):
        raise ValueError("neural adapter checkpoint state_dict must contain tensors")
    if not all(bool(torch.isfinite(value).all()) for value in state_dict.values()):
        raise ValueError("neural adapter checkpoint contains NaN or Inf")
    model = NeuralPhysicsAdapter(config).to(map_location)
    try:
        model.load_state_dict(dict(state_dict), strict=True)
    except RuntimeError as error:
        raise ValueError("neural adapter checkpoint weights do not match config") from error
    expected_count = payload.get("learned_parameter_count")
    if not isinstance(expected_count, int) or expected_count != model.parameter_count():
        raise ValueError("neural adapter checkpoint parameter count does not match")
    return model.eval()


def run_neural_adaptive_physics(
    *,
    training_steps: int = DEFAULT_TRAINING_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    evaluate_public_scenarios: bool = True,
) -> tuple[NeuralAdaptivePhysicsResult, NeuralPhysicsAdapter]:
    started = time.perf_counter()
    model, curve, training_seconds, changed_parameters = train_neural_physics_adapter(
        steps=training_steps,
        batch_size=batch_size,
    )
    development = evaluate_neural_physics_adapter(
        model,
        seed=DEFAULT_DEVELOPMENT_SEED,
    )
    compositional_ood = evaluate_neural_physics_adapter(
        model,
        seed=DEFAULT_OOD_SEED,
        ood=True,
    )
    scenarios: tuple[analytic_gate.AdaptivePhysicsScenarioResult, ...] = ()
    planning: tuple[state_gate.MultiContactPlanningResult, ...] = ()
    if evaluate_public_scenarios:
        scenario_values = []
        planning_inputs = None
        for object_count in (4, 6, 8):
            scenario, dynamics, belief, truth, truth_by_slot = analytic_gate._evaluate_scenario(
                object_count,
                parameter_calibrator=lambda current, target, mapping: _public_neural_calibrator(
                    model,
                    current,
                    target,
                    mapping,
                ),
            )
            scenario_values.append(scenario)
            if object_count == 8:
                planning_inputs = dynamics, belief, truth, truth_by_slot
        scenarios = tuple(scenario_values)
        assert planning_inputs is not None
        planning = tuple(
            visual_gate._evaluate_planning(candidate_count, *planning_inputs)
            for candidate_count in (8, 32)
        )

    learned_parameters = sum(parameter.numel() for parameter in model.parameters())
    nonzero_parameters = sum(
        int(torch.count_nonzero(parameter)) for parameter in model.parameters()
    )
    learned_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    failures = []
    if development.mean_relative_error > 0.08:
        failures.append("development_parameter_mean")
    if development.p95_relative_error > 0.20:
        failures.append("development_parameter_p95")
    if compositional_ood.mean_relative_error > 0.16:
        failures.append("ood_parameter_mean")
    if development.permutation_maximum_difference > 1.0e-6:
        failures.append("evidence_permutation")
    if nonzero_parameters / learned_parameters < 0.90:
        failures.append("learned_parameter_activity")
    if learned_bytes > 1 << 20:
        failures.append("learned_weight_budget")
    if curve[-1]["loss"] >= 0.20 * curve[0]["loss"]:
        failures.append("objective_reduction")
    for scenario in scenarios:
        prefix = f"n{scenario.object_count}"
        if scenario.final_parameter_errors["all"]["mean"] > 0.10:
            failures.append(f"{prefix}:parameter_mean")
        if scenario.endpoint_position_rmse_m > 0.12:
            failures.append(f"{prefix}:endpoint_position")
        if not scenario.finite or not scenario.source_unchanged:
            failures.append(f"{prefix}:invariant")
    for item in planning:
        if not item.winner_correct:
            failures.append(f"planning_k{item.candidate_count}:winner")
        if not item.serial_vectorized_parity or item.maximum_cost_difference > 1.0e-6:
            failures.append(f"planning_k{item.candidate_count}:parity")

    # The final checkpoint digest is filled by the publisher; this placeholder
    # keeps the pure run function independent of artifact paths.
    result = NeuralAdaptivePhysicsResult(
        schema=NEURAL_ADAPTIVE_PHYSICS_SCHEMA,
        protocol_sha256=neural_adaptive_protocol_sha256(
            training_steps=training_steps,
            batch_size=batch_size,
        ),
        training_steps=training_steps,
        training_batch_size=batch_size,
        training_examples=training_steps * batch_size,
        training_seconds=training_seconds,
        training_loss_initial=curve[0]["loss"],
        training_loss_final=curve[-1]["loss"],
        training_curve=curve,
        development=development,
        compositional_ood=compositional_ood,
        scenarios=scenarios,
        planning=planning,
        learned_parameter_count=learned_parameters,
        nonzero_learned_parameter_count=nonzero_parameters,
        changed_parameter_count=changed_parameters,
        learned_weight_bytes=learned_bytes,
        checkpoint_sha256="pending-publication",
        gate_failures=tuple(failures),
        passed=not failures,
        evaluation_seconds=time.perf_counter() - started,
    )
    return result, model


def _summary(
    result: NeuralAdaptivePhysicsResult,
    *,
    run_id: str,
    run_bytes: int,
    archive_bytes: int,
) -> CapabilityRunSummary:
    scenario_by_count = {item.object_count: item for item in result.scenarios}
    planning_by_k = {
        str(item.candidate_count): {
            "winner_accuracy": float(item.winner_correct),
            "median_normalized_regret": item.normalized_regret,
            "goal_success": float(item.goal_success),
            "vectorized_latency_seconds": item.vectorized_latency_seconds,
            "maximum_cost_difference": item.maximum_cost_difference,
        }
        for item in result.planning
    }
    neutral_error = (
        sum(item.initial_parameter_errors["all"]["mean"] for item in result.scenarios)
        / len(result.scenarios)
        if result.scenarios
        else result.development.prefix_mean_relative_error["free_motion"]
    )
    candidate_error = (
        sum(item.final_parameter_errors["all"]["mean"] for item in result.scenarios)
        / len(result.scenarios)
        if result.scenarios
        else result.development.mean_relative_error
    )
    if result.scenarios:
        parameter_stages = [
            {
                "stage": result.scenarios[0].parameter_stages[index]["stage"],
                **{
                    key: sum(item.parameter_stages[index][key] for item in result.scenarios)
                    / len(result.scenarios)
                    for key in (
                        "mean_relative_error",
                        "mass_relative_error",
                        "restitution_relative_error",
                        "drag_relative_error",
                        "friction_relative_error",
                    )
                },
            }
            for index in range(len(result.scenarios[0].parameter_stages))
        ]
    else:
        parameter_stages = [
            {"stage": "neutral structured prior", "mean_relative_error": neutral_error},
            {
                "stage": "learned development evidence",
                "mean_relative_error": result.development.mean_relative_error,
            },
        ]
    return CapabilityRunSummary(
        schema=CAPABILITY_SUMMARY_SCHEMA,
        run_id=run_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        lifecycle_status="completed" if result.passed else "failed",
        outcome="neural_learning_bridge_passed"
        if result.passed
        else "neural_learning_bridge_failed",
        source_format=NEURAL_ADAPTIVE_PHYSICS_SCHEMA,
        configuration={
            "single_model": True,
            "ensemble": False,
            "public_calibrated_rgbd": bool(result.scenarios),
            "planning_used_as_training_loss": False,
            "model_profile": {
                "name": "Single shared neural-adaptive structured world model",
                "variant": "2-layer evidence transformer · analytic rigid rollout",
                "stages": [
                    {
                        "role": "Observe",
                        "name": "Calibrated RGB-D",
                        "detail": "metric object histories",
                    },
                    {
                        "role": "Learn",
                        "name": "Evidence transformer",
                        "detail": "2 layers · 4 heads · width 48",
                    },
                    {
                        "role": "Adapt",
                        "name": "Physical belief",
                        "detail": "mass · drag · bounce · friction",
                    },
                    {
                        "role": "Predict",
                        "name": "Analytic rigid dynamics",
                        "detail": "learned belief · exact contacts",
                    },
                    {
                        "role": "Plan",
                        "name": "Batched action rollouts",
                        "detail": "planning remains evaluation-only",
                    },
                ],
                "evolution_note": (
                    "Transformer weights learned across streamed episodes; each new object "
                    "adapts through its public evidence tokens without online backpropagation."
                ),
            },
        },
        provenance={
            "protocol_sha256": result.protocol_sha256,
            "checkpoint_loaded": False,
            "checkpoint_sha256": result.checkpoint_sha256,
            "trained_from_scratch": True,
            "truth_parameters_used_only_as_training_targets": True,
            "truth_parameters_in_runtime_inputs": False,
            "runtime_adapter_inputs": (
                "public metric position transitions, known action impulses, boundary normals"
            ),
            "planning_used_as_training_loss": False,
        },
        scores={
            "candidate": {"value": candidate_error, "supported_weight": 1.0},
            "incumbent": {"value": neutral_error, "supported_weight": 1.0},
            "selected": "neural_physics_development_candidate",
        },
        factor_metrics={
            "learned_parameter_identification": {
                "status": "passed" if result.development.mean_relative_error <= 0.08 else "failed",
                "score": result.development.mean_relative_error,
                "p95_relative_error": result.development.p95_relative_error,
                **result.development.by_parameter,
            },
            "compositional_ood": {
                "status": "passed"
                if result.compositional_ood.mean_relative_error <= 0.16
                else "failed",
                "score": result.compositional_ood.mean_relative_error,
                "p95_relative_error": result.compositional_ood.p95_relative_error,
                **result.compositional_ood.by_parameter,
            },
            **{
                f"neural_public_rgbd_n{count}": {
                    "status": "passed"
                    if item.final_parameter_errors["all"]["mean"] <= 0.10
                    and item.endpoint_position_rmse_m <= 0.12
                    else "failed",
                    "parameter_relative_error": item.final_parameter_errors["all"]["mean"],
                    "four_second_position_rmse_m": item.endpoint_position_rmse_m,
                    "collision_f1": item.contact_pair_f1,
                }
                for count, item in scenario_by_count.items()
            },
        },
        cell_metrics={
            f"N{count}/neural_adaptive_physics/mixed_rigid": {
                "parameter_relative_error": {
                    "value": item.final_parameter_errors["all"]["mean"],
                    "support": 4 * count,
                },
                "four_second_position_rmse_m": {
                    "value": item.endpoint_position_rmse_m,
                    "support": count,
                },
            }
            for count, item in scenario_by_count.items()
        },
        horizon_curves={
            "training_objective": {
                str(int(item["step"])): item["loss"] for item in result.training_curve
            },
            **(
                {
                    "candidate_position_rmse_m": {
                        key: max(item.position_curve[key] for item in result.scenarios)
                        for key in result.scenarios[0].position_curve
                    }
                }
                if result.scenarios
                else {}
            ),
        },
        uncertainty={
            "status": "learned normalized diagonal variance; calibration remains development-only"
        },
        planning={
            "status": "passed"
            if result.planning and all(item.winner_correct for item in result.planning)
            else "unmeasured",
            "by_candidate_count": planning_by_k,
            "serial_vectorized_winner_parity": bool(result.planning)
            and all(item.serial_vectorized_parity for item in result.planning),
            "maximum_cost_difference": max(
                (item.maximum_cost_difference for item in result.planning),
                default=0.0,
            ),
        },
        resources={
            "evaluation_seconds": result.evaluation_seconds,
            "training_seconds": result.training_seconds,
            "learned_parameter_count": result.learned_parameter_count,
            "nonzero_learned_parameter_count": result.nonzero_learned_parameter_count,
            "changed_learned_parameter_count": result.changed_parameter_count,
            "learned_weight_bytes": result.learned_weight_bytes,
            "optimizer_updates": result.training_steps,
            "training_examples": result.training_examples,
            "online_adaptation_updates_accepted": sum(
                4 * item.object_count for item in result.scenarios
            ),
            "online_adaptation_updates_attempted": sum(
                4 * item.object_count for item in result.scenarios
            ),
            "adaptation_update_label": "observation-conditioned parameter adaptations",
            "adaptation_update_detail": "neural inference; no online optimizer",
            **(
                {
                    "n8_rollout_latency_seconds": scenario_by_count[8].rollout_latency_seconds,
                    "public_probe_frame_count": sum(
                        item.public_probe_frame_count for item in result.scenarios
                    ),
                    "k32_vectorization_speedup": next(
                        item.vectorization_speedup
                        for item in result.planning
                        if item.candidate_count == 32
                    ),
                }
                if 8 in scenario_by_count and result.planning
                else {}
            ),
        },
        artifacts={
            "run_bytes": run_bytes,
            "archive_bytes": archive_bytes,
            "checkpoint_sha256": result.checkpoint_sha256,
        },
        selection={
            "selected": "neural_physics_development_candidate",
            "promotion_evaluated": False,
            "gate_failures": list(result.gate_failures),
        },
        failure_attribution={
            "primary_bottleneck": result.gate_failures[0] if result.gate_failures else "none",
            "ablation_owner": "learned public-evidence physical identifier",
        },
        qualitative={
            "parameter_convergence": parameter_stages,
            "training_curve": list(result.training_curve),
            "forecast_animations": [item.animation for item in result.scenarios],
            "diagnostic_contact_sheets": [],
        },
        unsupported_claims=(
            "protected checkpoint promotion",
            "end-to-end learned pixels-to-futures",
            "learned deformable or articulated dynamics",
            "unknown camera calibration",
        ),
        scope_limitations=(
            "synthetic streamed optimizer data",
            "runtime adaptation uses short controlled evidence histories",
            "analytic rigid-body rollout remains authoritative",
            "no online gradient updates",
        ),
    ).validate()


def publish_neural_adaptive_physics(
    result: NeuralAdaptivePhysicsResult,
    model: NeuralPhysicsAdapter,
    *,
    run_directory: str | Path,
    runs_root: str | Path = "runs",
    archive_root: str | Path = ".archive",
) -> CapabilityRunSummary:
    run = Path(run_directory).expanduser().resolve()
    run.mkdir(parents=True, exist_ok=True)
    checkpoint = run / "model.pt"
    _atomic_torch_save(
        _checkpoint_payload(
            model,
            result=result,
        ),
        checkpoint,
    )
    result = replace(result, checkpoint_sha256=_sha256(checkpoint))
    atomic_write_text(
        run / "neural_adaptive_physics.json",
        json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    archive_bytes = int(inventory_runs(runs_root, archive_root=archive_root)["archive_bytes"])
    summary = _summary(result, run_id=run.name, run_bytes=0, archive_bytes=archive_bytes)
    for _ in range(8):
        write_capability_summary(summary, run / "capability_summary.json")
        write_run_report(summary, run)
        write_run_manifest(
            run,
            role="candidate",
            status="completed" if result.passed else "failed",
            artifacts={
                "capability_summary.json": "summary",
                "model.pt": "checkpoint" if result.passed else "rejected",
                "neural_adaptive_physics.json": "summary",
                "report.html": "report",
            },
        )
        actual = sum(
            path.stat().st_size
            for path in run.iterdir()
            if path.is_file() and not path.is_symlink()
        )
        if summary.artifacts["run_bytes"] == actual:
            break
        summary = replace(summary, artifacts={**summary.artifacts, "run_bytes": actual})
    else:
        raise RuntimeError("neural-adaptive evidence byte count did not converge")
    build_progress_dashboard(runs_root, archive_root=archive_root)
    return summary


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_TRAINING_STEPS",
    "NEURAL_ADAPTIVE_PHYSICS_SCHEMA",
    "NeuralAdaptivePhysicsResult",
    "NeuralPhysicsMetrics",
    "SyntheticPhysicsBatch",
    "evaluate_neural_physics_adapter",
    "load_neural_physics_adapter",
    "neural_adaptive_protocol_sha256",
    "publish_neural_adaptive_physics",
    "run_neural_adaptive_physics",
    "synthetic_physics_batch",
    "train_neural_physics_adapter",
]
