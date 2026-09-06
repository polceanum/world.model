"""Bounded specification-1.61 dynamic-set training engine.

The materializer deliberately stops at accepted public episodes, while the
objective module deliberately starts at already-aligned tensors.  This module
does not invent the missing alignment policy.  Instead it defines a narrow,
checkpointable adapter boundary and owns everything around that boundary:

* the exact six-by-B4, 22-cell random-access draw schedule;
* construction of the frozen physical objective (never a planning objective);
* causal censoring audits for hidden versus public known impulses;
* strict set-proposer/relation-only AdamW ownership;
* pre-mutation gradient/state checks and transactional rejected updates;
* 512-update validation cadence, the exact 64-example disposable screen; and
* exact model/optimizer/scheduler/adapter/RNG/next-draw resume payloads.

The concrete episode-to-objective adapter is intentionally experiment-owned.
It must align public RGB-D observations and labels without exposing simulator
truth to :class:`~world_model.runtime.online_world_model.OnlineWorldModel`.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from numbers import Real
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor, nn

from world_model.training.dynamic_set_campaign import (
    DEFAULT_CAMPAIGN,
    DisposableScreenMetrics,
    DynamicSetCampaignConfig,
    disposable_screen_failures,
)
from world_model.training.dynamic_set_materializer import materialize_dynamic_set_episode
from world_model.training.dynamic_set_objectives import (
    DEFAULT_DYNAMIC_SET_LOSS_WEIGHTS,
    DynamicSetLosses,
    DynamicSetLossWeights,
    DynamicsLossInputs,
    PerceptionLossInputs,
    dynamic_set_objective,
    dynamic_set_objective_regret,
)
from world_model.training.dynamic_set_optimization import (
    DEFAULT_DYNAMIC_SET_OPTIMIZER_CONFIG,
    DynamicSetOptimizerConfig,
    build_dynamic_set_optimizer,
)
from world_model.training.dynamic_set_optimization import (
    learning_rate_multiplier as optimizer_learning_rate_multiplier,
)
from world_model.training.dynamic_set_protocol import (
    EXAMPLES_PER_UPDATE,
    MICROBATCH_SIZE,
    MICROBATCHES_PER_UPDATE,
    PHYSICAL_CELL_COUNT,
    PHYSICAL_CELLS,
    PHYSICAL_SPLIT_SIZES,
    SIMULATOR_VERSION,
    SPECIFICATION_VERSION,
    PhysicalManifestRow,
    canonical_sha256,
    physical_manifest,
)
from world_model.training.dynamic_set_sampling import DynamicSetMicrobatchSchedule
from world_model.training.dynamic_set_scene import DYNAMIC_SET_FRAMES
from world_model.training.qualification_core import validated_sha256

CHECKPOINT_SCHEMA = "dynamic_set_trainer_v3"
SCREEN_EXAMPLE_COUNT = 64


def _screen_rows() -> tuple[PhysicalManifestRow, ...]:
    """Return the exact all-regime screen without opening any protected split.

    Three complete 22-cell cycles contain both contact causal regimes and
    every dynamic lifecycle schedule.  We remove the third, redundant
    no-action sample from two static/no-contact cells to obtain exactly 64
    rows while preserving action/no-action support in every physical cell.
    """

    source = physical_manifest("training")[: 3 * PHYSICAL_CELL_COUNT]
    omitted_ordinals = {2 * PHYSICAL_CELL_COUNT, 2 * PHYSICAL_CELL_COUNT + 2}
    rows = tuple(row for row in source if row.ordinal not in omitted_ordinals)
    if len(rows) != SCREEN_EXAMPLE_COUNT or len({row.ordinal for row in rows}) != len(rows):
        raise RuntimeError("dynamic-set screen must contain exactly 64 unique training rows")
    by_cell = {
        index: tuple(row for row in rows if row.cell_index == index)
        for index in range(PHYSICAL_CELL_COUNT)
    }
    if any(not values for values in by_cell.values()):
        raise RuntimeError("dynamic-set screen must cover every physical cell")
    for index, cell in enumerate(PHYSICAL_CELLS):
        values = by_cell[index]
        if {row.known_action for row in values} != {False, True}:
            raise RuntimeError("dynamic-set screen must cover action/no-action in every cell")
        if cell.contact and {(row.known_action, row.contact_origin) for row in values} != {
            (False, "natural"),
            (True, "action_induced"),
        }:
            raise RuntimeError("dynamic-set screen must cover both contact causal regimes")
        if cell.contact and {row.contact_geometry for row in values} != {
            "head_on",
            "glancing",
        }:
            raise RuntimeError("dynamic-set screen must cover both contact geometries")
        if cell.dynamic_membership and {row.lifecycle_schedule for row in values} != {
            "birth",
            "removal",
            "remove_then_birth",
        }:
            raise RuntimeError("dynamic-set screen must cover every lifecycle schedule")
    return rows


SCREEN_ROWS: tuple[PhysicalManifestRow, ...] = _screen_rows()
SCREEN_MANIFEST_SHA256 = "b6c3deb2c8fb2fa9d58d5cf29d38f695de3bc93430a02dd47cdcca38add3db0b"
if canonical_sha256([asdict(row) for row in SCREEN_ROWS]) != SCREEN_MANIFEST_SHA256:
    raise RuntimeError("dynamic-set screen manifest differs from its source freeze")
MINIMUM_COMPLETE_GRADIENT_RETENTION = 0.10
DEFAULT_MAXIMUM_GRADIENT_NORM = 1.0


@dataclass(frozen=True)
class DynamicSetCausalSupport:
    """Public action evidence and target times for one aligned rollout batch.

    ``externally_actuated`` and ``known_action_observed`` cover the complete
    source episode and have shape ``[B,F,N]``.  ``anchor_frame_index`` has
    shape ``[B]`` and ``target_frame_index`` has shape ``[B,T]``, matching the
    time axis returned in :class:`DynamicsLossInputs`.
    """

    externally_actuated: Tensor
    known_action_observed: Tensor
    anchor_frame_index: Tensor
    target_frame_index: Tensor


@dataclass(frozen=True)
class DynamicSetTrainingMicrobatch:
    """The four exact manifest rows and their accepted materializations."""

    update_index: int
    microbatch_index: int
    dataset_indices: tuple[int, ...]
    rows: tuple[PhysicalManifestRow, ...]
    materializations: tuple[object, ...]


@dataclass(frozen=True)
class DynamicSetObjectiveInputs:
    """Only values an adapter may return to the optimizer engine."""

    perception: PerceptionLossInputs
    dynamics: DynamicsLossInputs
    causal_support: DynamicSetCausalSupport


@runtime_checkable
class DynamicSetObjectiveAdapter(Protocol):
    """Strict, checkpointable episode-to-objective adapter contract.

    Implementations may run the public model and align public supervision, but
    may not return a scalar loss.  The engine alone calls
    :func:`dynamic_set_objective`, which makes planning, winner, regret,
    ranking, and task-success losses structurally impossible.
    """

    def build_objective_inputs(
        self,
        model: nn.Module,
        microbatch: DynamicSetTrainingMicrobatch,
    ) -> DynamicSetObjectiveInputs: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


@dataclass(frozen=True)
class DynamicSetCheckpointBindings:
    manifest_sha256: str
    protocol_sha256: str
    config_sha256: str
    source_sha256: str

    def validate(self) -> DynamicSetCheckpointBindings:
        for item in fields(self):
            validated_sha256(getattr(self, item.name), label=item.name)
        return self


@dataclass(frozen=True)
class DynamicSetValidationRecord:
    completed_updates: int
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class DynamicSetUpdateReport:
    completed_updates: int
    next_update_index: int
    materialized_examples: int
    cell_counts: tuple[int, ...]
    objective: float
    loss_terms: Mapping[str, float]
    perception_gradient_norm: float
    relation_gradient_norm: float
    raw_gradient_norm: float
    applied_gradient_norm: float
    complete_gradient_retention: float
    finite_owner_gradients: bool
    validation: DynamicSetValidationRecord | None


@dataclass(frozen=True)
class DynamicSetScreenSnapshot:
    """One read-only evaluation over the exact disposable screen population."""

    example_count: int
    optimization_objective: float
    proposal_f1: float
    collision_f1: float

    @property
    def objective_regret(self) -> float:
        """Return the nonnegative diagnostic; training still uses the signed loss."""

        return dynamic_set_objective_regret(self.optimization_objective)


@dataclass(frozen=True)
class DynamicSetScreenReport:
    metrics: DisposableScreenMetrics
    example_count: int
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


class DynamicSetUpdateRejected(RuntimeError):
    """A failed update that left every mutable training authority unchanged."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


Materializer = Callable[[PhysicalManifestRow], object]
ValidationHook = Callable[[nn.Module, int], Mapping[str, float]]
ScreenHook = Callable[[nn.Module, int], DynamicSetScreenSnapshot]
RuntimeReset = Callable[[nn.Module, int], None]


def dynamic_set_perception_frame_index(update_index: int, microbatch_index: int) -> int:
    """Return the stateless rotating perception frame for one B4 microbatch."""

    for name, value in (("update_index", update_index), ("microbatch_index", microbatch_index)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if microbatch_index >= MICROBATCHES_PER_UPDATE:
        raise ValueError("microbatch_index lies outside the six-B4 update")
    return (update_index * MICROBATCHES_PER_UPDATE + microbatch_index) % DYNAMIC_SET_FRAMES


def _configuration_payload(config: object) -> Mapping[str, Any]:
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
    elif isinstance(config, Mapping):
        value = deepcopy(dict(config))
    else:
        raise TypeError("resolved_config must be a mapping or expose to_dict()")
    if not isinstance(value, Mapping):
        raise TypeError("resolved configuration payload must be a mapping")
    # Canonicalization is also the strict JSON-native validation step.
    canonical_sha256(dict(value))
    return deepcopy(dict(value))


def dynamic_set_training_protocol_sha256(
    *,
    optimizer_config: DynamicSetOptimizerConfig = DEFAULT_DYNAMIC_SET_OPTIMIZER_CONFIG,
    campaign_config: DynamicSetCampaignConfig = DEFAULT_CAMPAIGN,
    loss_weights: DynamicSetLossWeights = DEFAULT_DYNAMIC_SET_LOSS_WEIGHTS,
    maximum_gradient_norm: float = DEFAULT_MAXIMUM_GRADIENT_NORM,
    minimum_complete_gradient_retention: float = MINIMUM_COMPLETE_GRADIENT_RETENTION,
) -> str:
    """Hash every optimizer-engine semantic that can change a continuation."""

    optimizer_config.validate()
    campaign_config.validate()
    loss_weights.validate()
    for name, value in (
        ("maximum_gradient_norm", maximum_gradient_norm),
        ("minimum_complete_gradient_retention", minimum_complete_gradient_retention),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{name} must be a finite real number")
    if maximum_gradient_norm <= 0.0:
        raise ValueError("maximum_gradient_norm must be positive")
    if not 0.0 < minimum_complete_gradient_retention <= 1.0:
        raise ValueError("minimum_complete_gradient_retention must lie in (0,1]")
    payload = {
        "schema": "dynamic_set_training_protocol_v1",
        "specification_version": SPECIFICATION_VERSION,
        "simulator_version": SIMULATOR_VERSION,
        "physical_cells": [asdict(cell) for cell in PHYSICAL_CELLS],
        "physical_cell_count": PHYSICAL_CELL_COUNT,
        "microbatch_size": MICROBATCH_SIZE,
        "microbatches_per_update": MICROBATCHES_PER_UPDATE,
        "examples_per_update": EXAMPLES_PER_UPDATE,
        "perception_frame_count": DYNAMIC_SET_FRAMES,
        "perception_frame_schedule": "(update*6+microbatch)%56",
        "contextual_materialization": "certified_compact_trace_plus_exact_selected_render_v1",
        "optimizer": asdict(optimizer_config),
        "campaign": asdict(campaign_config),
        "loss_weights": asdict(loss_weights),
        "loss_schema": [item.name for item in fields(DynamicSetLosses)],
        "maximum_gradient_norm": float(maximum_gradient_norm),
        "minimum_complete_gradient_retention": float(minimum_complete_gradient_retention),
        "screen_example_count": SCREEN_EXAMPLE_COUNT,
    }
    return canonical_sha256(payload)


def make_dynamic_set_checkpoint_bindings(
    rows: Sequence[PhysicalManifestRow],
    *,
    resolved_config: object,
    source_provenance: Mapping[str, Any],
    optimizer_config: DynamicSetOptimizerConfig = DEFAULT_DYNAMIC_SET_OPTIMIZER_CONFIG,
    campaign_config: DynamicSetCampaignConfig = DEFAULT_CAMPAIGN,
    loss_weights: DynamicSetLossWeights = DEFAULT_DYNAMIC_SET_LOSS_WEIGHTS,
    maximum_gradient_norm: float = DEFAULT_MAXIMUM_GRADIENT_NORM,
    minimum_complete_gradient_retention: float = MINIMUM_COMPLETE_GRADIENT_RETENTION,
) -> DynamicSetCheckpointBindings:
    """Return exact manifest/protocol/config/source checkpoint bindings."""

    if not isinstance(source_provenance, Mapping) or not source_provenance:
        raise ValueError("source_provenance must be a nonempty mapping")
    manifest_payload = [asdict(row) for row in rows]
    config_payload = _configuration_payload(resolved_config)
    # Validate the complete source payload rather than trusting one optionally
    # absent Git field.  Formal callers should pass capture_git_metadata().
    source_payload = deepcopy(dict(source_provenance))
    return DynamicSetCheckpointBindings(
        manifest_sha256=canonical_sha256(manifest_payload),
        protocol_sha256=dynamic_set_training_protocol_sha256(
            optimizer_config=optimizer_config,
            campaign_config=campaign_config,
            loss_weights=loss_weights,
            maximum_gradient_norm=maximum_gradient_norm,
            minimum_complete_gradient_retention=minimum_complete_gradient_retention,
        ),
        config_sha256=canonical_sha256(dict(config_payload)),
        source_sha256=canonical_sha256(source_payload),
    ).validate()


def causal_scene_predictable_mask(support: DynamicSetCausalSupport) -> Tensor:
    """Return exact ``[B,T]`` deterministic support after public actions.

    Known actions are public causal inputs and therefore do not censor their
    future.  Any externally applied impulse not marked public censors the
    complete coupled scene from its occurrence through the requested target.
    """

    if not isinstance(support, DynamicSetCausalSupport):
        raise TypeError("causal support must be DynamicSetCausalSupport")
    external = support.externally_actuated
    known = support.known_action_observed
    anchors = support.anchor_frame_index
    targets = support.target_frame_index
    if not isinstance(external, Tensor) or external.dtype is not torch.bool or external.ndim != 3:
        raise TypeError("externally_actuated must be boolean [B,F,N]")
    if (
        not isinstance(known, Tensor)
        or known.dtype is not torch.bool
        or known.shape != external.shape
        or known.device != external.device
    ):
        raise TypeError("known_action_observed must match externally_actuated")
    if bool((known & ~external).any()):
        raise ValueError("public known actions must be a subset of external actuation")
    batch, frames, _ = external.shape
    for name, value, shape in (
        ("anchor_frame_index", anchors, (batch,)),
        ("target_frame_index", targets, None),
    ):
        if not isinstance(value, Tensor) or value.dtype is not torch.int64:
            raise TypeError(f"{name} must be an int64 tensor")
        if value.device != external.device:
            raise ValueError(f"{name} must share the action-evidence device")
        if shape is not None and value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if targets.ndim != 2 or targets.shape[0] != batch or targets.shape[1] == 0:
        raise ValueError("target_frame_index must have nonempty shape [B,T]")
    if bool((anchors < 0).any()) or bool((anchors >= frames - 1).any()):
        raise ValueError("anchor frame indices lie outside the episode")
    if bool((targets <= anchors[:, None]).any()) or bool((targets >= frames).any()):
        raise ValueError("target frames must strictly follow their anchors inside the episode")
    if targets.shape[1] > 1 and bool((targets[:, 1:] <= targets[:, :-1]).any()):
        raise ValueError("target frame indices must increase strictly")

    hidden_by_frame = (external & ~known).any(dim=-1)
    frame_index = torch.arange(frames, device=external.device)
    inside = (frame_index[None, None, :] > anchors[:, None, None]) & (
        frame_index[None, None, :] <= targets[:, :, None]
    )
    hidden_before_target = (hidden_by_frame[:, None, :] & inside).any(dim=-1)
    return ~hidden_before_target


def _assert_finite_tree(value: Any, *, root: str) -> None:
    pending: list[tuple[str, Any]] = [(root, value)]
    while pending:
        name, item = pending.pop()
        if isinstance(item, Tensor):
            if (item.is_floating_point() or item.is_complex()) and not bool(
                torch.isfinite(item).all()
            ):
                raise FloatingPointError(f"{name} contains NaN or Inf")
        elif isinstance(item, Mapping):
            pending.extend((f"{name}.{key}", child) for key, child in item.items())
        elif isinstance(item, (list, tuple)):
            pending.extend((f"{name}[{index}]", child) for index, child in enumerate(item))
        elif (
            isinstance(item, Real) and not isinstance(item, bool) and not math.isfinite(float(item))
        ):
            raise FloatingPointError(f"{name} contains a nonfinite scalar")


def _clone_model_state(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def dynamic_set_model_state_sha256(model_state: Mapping[str, Any]) -> str:
    """Hash exact tensor names, layouts, dtypes, and bytes."""

    if not isinstance(model_state, Mapping):
        raise TypeError("model_state must be a mapping")
    digest = hashlib.sha256()
    for name in sorted(model_state):
        if type(name) is not str or not name:
            raise ValueError("model-state names must be nonempty strings")
        value = model_state[name]
        if not isinstance(value, Tensor):
            raise TypeError(f"model state entry {name!r} is not a tensor")
        tensor = value.detach().to(device="cpu").contiguous()
        header = json.dumps(
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, byteorder="big"))
        digest.update(header)
        raw = tensor.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, byteorder="big"))
        digest.update(raw)
    return digest.hexdigest()


def _assert_state_equal(
    actual: Mapping[str, Tensor],
    expected: Mapping[str, Tensor],
    *,
    excluded: frozenset[str] = frozenset(),
) -> None:
    if set(actual) != set(expected):
        raise RuntimeError("model state schema changed during an optimizer update")
    for name, before in expected.items():
        if name in excluded:
            continue
        after = actual[name]
        if (
            after.shape != before.shape
            or after.dtype != before.dtype
            or not torch.equal(after, before)
        ):
            raise RuntimeError(f"model state {name!r} mutated outside AdamW ownership")


def _trees_equal(actual: Any, expected: Any) -> bool:
    """Return exact equality for checkpointable transactional state."""

    if isinstance(actual, Tensor) or isinstance(expected, Tensor):
        return (
            isinstance(actual, Tensor)
            and isinstance(expected, Tensor)
            and actual.shape == expected.shape
            and actual.dtype == expected.dtype
            and actual.device == expected.device
            and torch.equal(actual, expected)
        )
    if isinstance(actual, Mapping) or isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and isinstance(expected, Mapping)
            and set(actual) == set(expected)
            and all(_trees_equal(actual[key], expected[key]) for key in actual)
        )
    if isinstance(actual, (tuple, list)) or isinstance(expected, (tuple, list)):
        return (
            type(actual) is type(expected)
            and len(actual) == len(expected)
            and all(_trees_equal(left, right) for left, right in zip(actual, expected, strict=True))
        )
    return type(actual) is type(expected) and bool(actual == expected)


def _capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": deepcopy(random.getstate()),
        # Keep the checkpoint compatible with ``torch.load(weights_only=True)``:
        # NumPy ndarrays otherwise require an unsafe pickle global.
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.as_tensor(numpy_state[1].astype(np.int64, copy=True)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": (
            [value.clone() for value in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise ValueError("RNG state has an incompatible schema")
    torch_cpu = state["torch_cpu"]
    if (
        not isinstance(torch_cpu, Tensor)
        or torch_cpu.dtype is not torch.uint8
        or torch_cpu.ndim != 1
    ):
        raise ValueError("torch CPU RNG state is invalid")
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, Mapping) or set(numpy_state) != {
        "bit_generator",
        "state",
        "position",
        "has_gauss",
        "cached_gaussian",
    }:
        raise ValueError("NumPy RNG state is invalid")
    numpy_values = numpy_state["state"]
    if (
        type(numpy_state["bit_generator"]) is not str
        or not isinstance(numpy_values, Tensor)
        or numpy_values.dtype is not torch.int64
        or numpy_values.ndim != 1
        or isinstance(numpy_state["position"], bool)
        or not isinstance(numpy_state["position"], int)
        or numpy_state["has_gauss"] not in {0, 1}
        or isinstance(numpy_state["cached_gaussian"], bool)
        or not isinstance(numpy_state["cached_gaussian"], Real)
        or not math.isfinite(float(numpy_state["cached_gaussian"]))
    ):
        raise ValueError("NumPy RNG state is invalid")
    random.setstate(state["python"])
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            numpy_values.cpu().numpy().astype(np.uint32, copy=True),
            numpy_state["position"],
            numpy_state["has_gauss"],
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(torch_cpu.cpu())
    cuda_state = state["torch_cuda"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state on a CPU-only runtime")
        if not isinstance(cuda_state, list) or not all(
            isinstance(item, Tensor) for item in cuda_state
        ):
            raise ValueError("torch CUDA RNG state is invalid")
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_state])


def _numeric_metrics(value: Mapping[str, Any], *, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    result: dict[str, float] = {}
    for name, metric in value.items():
        if type(name) is not str or not name:
            raise ValueError(f"{label} names must be nonempty strings")
        if (
            isinstance(metric, bool)
            or not isinstance(metric, Real)
            or not math.isfinite(float(metric))
        ):
            raise ValueError(f"{label}.{name} must be a finite scalar")
        result[name] = float(metric)
    return result


def _materialized_episode(value: object) -> Mapping[str, Any]:
    episode = getattr(value, "episode", None)
    if not isinstance(episode, Mapping):
        raise TypeError("materializer results must expose an episode mapping")
    return episode


def _stack_public_action_evidence(
    microbatch: DynamicSetTrainingMicrobatch,
) -> tuple[Tensor, Tensor]:
    external_rows: list[Tensor] = []
    known_rows: list[Tensor] = []
    for row, materialization in zip(
        microbatch.rows,
        microbatch.materializations,
        strict=True,
    ):
        episode = _materialized_episode(materialization)
        events = episode.get("events")
        if not isinstance(events, Mapping):
            raise ValueError("materialized episode must contain events")
        external = events.get("externally_actuated")
        known = events.get("known_action_observed")
        if (
            not isinstance(external, Tensor)
            or external.dtype is not torch.bool
            or external.ndim != 2
        ):
            raise TypeError("episode externally_actuated must be boolean [F,N]")
        if (
            not isinstance(known, Tensor)
            or known.dtype is not torch.bool
            or known.shape != external.shape
            or known.device != external.device
        ):
            raise TypeError("episode known_action_observed must match externally_actuated")
        if bool((known & ~external).any()):
            raise ValueError("episode known actions must be a subset of external actuation")
        if int(known.sum()) != int(row.known_action):
            raise ValueError("episode public known-action count differs from its manifest row")
        external_rows.append(external)
        known_rows.append(known)
    try:
        return torch.stack(external_rows), torch.stack(known_rows)
    except RuntimeError as error:
        raise ValueError("materialized action-evidence shapes differ within B4") from error


def _validate_objective_inputs(
    value: DynamicSetObjectiveInputs,
    microbatch: DynamicSetTrainingMicrobatch,
) -> None:
    if not isinstance(value, DynamicSetObjectiveInputs):
        raise TypeError("objective adapter must return DynamicSetObjectiveInputs")
    perception = value.perception
    dynamics = value.dynamics
    if not isinstance(perception, PerceptionLossInputs) or not isinstance(
        dynamics, DynamicsLossInputs
    ):
        raise TypeError("objective adapter returned invalid physical input types")
    if perception.mask_logits.ndim != 4 or perception.mask_logits.shape[:2] != (
        MICROBATCH_SIZE,
        9,
    ):
        raise ValueError("set perception objective must use B4 and eight proposals plus background")
    if perception.appearance.shape != (MICROBATCH_SIZE, 8, 8):
        raise ValueError("set perception appearance objective must have shape [4,8,8]")
    if dynamics.predicted_state.ndim != 4 or dynamics.predicted_state.shape[0] != MICROBATCH_SIZE:
        raise ValueError("dynamic-set state objective must begin with B4")
    batch, steps, objects = dynamics.predicted_state.shape[:3]
    if objects != 6:
        raise ValueError("dynamic-set state objective must use six object slots")
    support = value.causal_support
    public_external, public_known = _stack_public_action_evidence(microbatch)
    if not torch.equal(support.externally_actuated, public_external):
        raise ValueError("adapter external-actuation evidence differs from public episodes")
    if not torch.equal(support.known_action_observed, public_known):
        raise ValueError("adapter known-action evidence differs from public episodes")
    predictable = causal_scene_predictable_mask(support)
    if predictable.shape != (batch, steps):
        raise ValueError("causal target grid must match the dynamics time axis")
    state_allowed = predictable.unsqueeze(-1).expand(batch, steps, objects)
    if bool((dynamics.contact_window_mask & ~state_allowed).any()):
        raise ValueError("contact point loss includes a future hidden impulse")
    if bool((dynamics.known_action_predictable_mask & ~state_allowed).any()):
        raise ValueError("known-action point loss includes a future hidden impulse")
    pair_allowed = predictable[:, :, None, None].expand(
        batch,
        steps,
        objects,
        objects,
    )
    if bool((dynamics.collision_support & ~pair_allowed).any()):
        raise ValueError("collision loss includes a future hidden impulse")


def _validate_training_rows(
    rows: Sequence[PhysicalManifestRow],
    *,
    screen_only: bool,
    test_only_synthetic_manifest: bool,
) -> tuple[PhysicalManifestRow, ...]:
    resolved = tuple(rows)
    if not resolved:
        raise ValueError("training_rows must not be empty")
    for row in resolved:
        if not isinstance(row, PhysicalManifestRow) or row.split != "training":
            raise ValueError("optimizer rows must be PhysicalManifestRow training rows")
        if row.cell_index not in range(PHYSICAL_CELL_COUNT):
            raise ValueError("training row has an invalid physical cell")
        cell = PHYSICAL_CELLS[row.cell_index]
        if (row.object_count, row.contact, row.dynamic_membership) != (
            cell.object_count,
            cell.contact,
            cell.dynamic_membership,
        ):
            raise ValueError("training row disagrees with its physical cell")
    if screen_only:
        if resolved != SCREEN_ROWS:
            raise ValueError("a disposable-screen trainer requires the exact frozen 64-row cover")
        if canonical_sha256([asdict(row) for row in resolved]) != SCREEN_MANIFEST_SHA256:
            raise AssertionError("frozen disposable-screen manifest digest is inconsistent")
    elif not test_only_synthetic_manifest:
        frozen = physical_manifest("training")
        if resolved != frozen:
            raise ValueError("production training requires the exact frozen 66,000-row manifest")
        if len(resolved) != PHYSICAL_SPLIT_SIZES["training"]:
            raise AssertionError("frozen training manifest size is inconsistent")
    return resolved


def _resolve_online_world_model_owners(
    model: nn.Module, config: object
) -> tuple[nn.Module, nn.Module, tuple[nn.Module, ...]]:
    from world_model.runtime.online_world_model import OnlineWorldModel

    if not isinstance(model, OnlineWorldModel):
        raise TypeError("from_online_world_model requires an OnlineWorldModel")
    config_payload = _configuration_payload(config)
    try:
        device = config_payload["device"]
        simulator = config_payload["simulator"]
        model_config = config_payload["model"]
        training = config_payload["training"]
        state = model_config["state"]
        rgb = model_config["rgb"]
        rgbd = model_config["rgbd"]
        dynamics = model_config["dynamics"]
        filter_config = model_config["filter"]
        lifecycle = model_config["lifecycle"]
    except (KeyError, TypeError) as error:
        raise ValueError("resolved configuration lacks the dynamic-set profile") from error
    expected = (
        device["preference"] == "cpu",
        device["cuda_amp"] is False,
        device["compile"] is False,
        simulator["image_size"] == [64, 64],
        simulator["frame_rate"] == 20,
        simulator["physics_rate"] == 120,
        simulator["sequence_frames"] == 56,
        simulator["min_objects"] == 1,
        simulator["max_objects"] == 6,
        simulator["world_bounds"] == [[-30.0, 30.0], [-30.0, 30.0], [-30.0, 30.0]],
        simulator["gravity"] == [0.0, 0.0, 0.0],
        simulator["radius_range"] == [0.21, 0.21],
        simulator["mass_range"] == [1.0, 1.0],
        simulator["restitution_range"] == [0.7, 0.7],
        simulator["drag_range"] == [0.05, 0.05],
        simulator["friction_range"] == [0.2, 0.2],
        simulator["known_camera_pose"] is True,
        simulator["render_noise_std"] == 0.0,
        simulator["external_impulse_probability"] == 0.0,
        model_config["max_objects"] == 6,
        state["geometry_dim"] == 1,
        state["appearance_dim"] == 8,
        state["residual_dynamics_dim"] == 1,
        state["modal_count"] == 0,
        state["fast_log_variance_min"] == -32.0,
        state["fast_log_variance_max"] == 6.0,
        rgb["enabled"] is False,
        rgbd["enabled"] is True,
        rgbd["observation_mode"] == "set",
        rgbd["max_objects"] == 6,
        rgbd["proposal_count"] == 8,
        (rgbd["set_feature_dim"], dynamics["relation_hidden_dim"])
        in {(32, None), (64, None), (32, 64)},
        rgbd["set_log_variance_residual_limit"] == 20.0,
        rgbd["world_radius"] == 0.21,
        rgbd["linear_drag"] == 0.05,
        rgbd["measurement_position_variance"] == 6.4e-5,
        rgbd["temporal_history_size"] == 16,
        rgbd["temporal_min_samples"] == 3,
        rgbd["temporal_velocity_variance_floor"] == 1.0e-14,
        dynamics["hidden_dim"] == 16,
        dynamics["max_substep"] == 1.0 / 120.0,
        dynamics["modal_dynamics_enabled"] is False,
        dynamics["continuous_pair_force_enabled"] is False,
        dynamics["node_acceleration_enabled"] is False,
        dynamics["event_driven_state_only_enabled"] is True,
        dynamics["relation_process_uncertainty_enabled"] is True,
        dynamics["process_noise_position"] == 1.0e-14,
        dynamics["process_noise_velocity"] == 1.0e-14,
        dynamics["attention_residual_enabled"] is False,
        dynamics["analytic_free_motion_only"] is False,
        filter_config["min_log_variance"] == -32.0,
        filter_config["max_log_variance"] == 8.0,
        lifecycle["birth_confirmations"] == 2,
        lifecycle["max_missed_steps"] == 1,
        training["batch_size"] == MICROBATCH_SIZE,
        training["steps"] == DEFAULT_CAMPAIGN.maximum_updates,
        training["learning_rate"] == DEFAULT_DYNAMIC_SET_OPTIMIZER_CONFIG.perception_learning_rate,
        training["weight_decay"] == DEFAULT_DYNAMIC_SET_OPTIMIZER_CONFIG.weight_decay,
        training["train_episodes"] == PHYSICAL_SPLIT_SIZES["training"],
        training["validation_episodes"] == PHYSICAL_SPLIT_SIZES["development"],
        training["fixed_dataset"] is True,
        training["eval_every"] == DEFAULT_CAMPAIGN.validation_interval_updates,
    )
    if not all(expected):
        raise ValueError("OnlineWorldModel/config is not the specification-1.61 CPU set profile")
    try:
        perception = model.observation_modules["rgbd"].set_proposer
        relation = model.dynamics.interactions.edge_network
    except (AttributeError, KeyError) as error:
        raise ValueError("OnlineWorldModel lacks the set-proposer/relation owners") from error
    if not isinstance(perception, nn.Module) or not isinstance(relation, nn.Module):
        raise ValueError("dynamic-set optimizer owners must be modules")
    try:
        legacy_modules = (
            model.dynamics.interactions.node_network,
            model.dynamics.uncertainty.process_network,
        )
    except AttributeError as error:
        raise ValueError("OnlineWorldModel lacks the explicit frozen legacy registry") from error
    if any(not isinstance(module, nn.Module) for module in legacy_modules):
        raise ValueError("dynamic-set legacy owners must be modules")
    return perception, relation, legacy_modules


class DynamicSetTrainer:
    """Transactional CPU trainer for the specification-1.61 physical objective."""

    def __init__(
        self,
        *,
        model: nn.Module,
        perception_owner: nn.Module,
        relation_owner: nn.Module,
        legacy_owners: Sequence[nn.Module],
        training_rows: Sequence[PhysicalManifestRow],
        objective_adapter: DynamicSetObjectiveAdapter,
        resolved_config: object,
        source_provenance: Mapping[str, Any],
        schedule_seed: int,
        materializer: Materializer = materialize_dynamic_set_episode,
        validation_hook: ValidationHook | None = None,
        runtime_reset: RuntimeReset | None = None,
        optimizer_config: DynamicSetOptimizerConfig = DEFAULT_DYNAMIC_SET_OPTIMIZER_CONFIG,
        campaign_config: DynamicSetCampaignConfig = DEFAULT_CAMPAIGN,
        loss_weights: DynamicSetLossWeights = DEFAULT_DYNAMIC_SET_LOSS_WEIGHTS,
        maximum_gradient_norm: float = DEFAULT_MAXIMUM_GRADIENT_NORM,
        minimum_complete_gradient_retention: float = MINIMUM_COMPLETE_GRADIENT_RETENTION,
        screen_only: bool = False,
        test_only_synthetic_manifest: bool = False,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an nn.Module")
        if not isinstance(objective_adapter, DynamicSetObjectiveAdapter):
            raise TypeError("objective_adapter must implement the checkpointable adapter protocol")
        if isinstance(schedule_seed, bool) or not isinstance(schedule_seed, int):
            raise TypeError("schedule_seed must be an integer")
        if not callable(materializer):
            raise TypeError("materializer must be callable")
        if validation_hook is not None and not callable(validation_hook):
            raise TypeError("validation_hook must be callable")
        if runtime_reset is not None and not callable(runtime_reset):
            raise TypeError("runtime_reset must be callable")
        if type(test_only_synthetic_manifest) is not bool:
            raise TypeError("test_only_synthetic_manifest must be an explicit boolean")
        if type(screen_only) is not bool:
            raise TypeError("screen_only must be an explicit boolean")
        self.optimizer_config = optimizer_config.validate()
        self.campaign_config = campaign_config.validate()
        self.loss_weights = loss_weights.validate()
        if (
            isinstance(maximum_gradient_norm, bool)
            or not isinstance(maximum_gradient_norm, Real)
            or not math.isfinite(float(maximum_gradient_norm))
            or maximum_gradient_norm <= 0.0
        ):
            raise ValueError("maximum_gradient_norm must be finite and positive")
        if (
            isinstance(minimum_complete_gradient_retention, bool)
            or not isinstance(minimum_complete_gradient_retention, Real)
            or not math.isfinite(float(minimum_complete_gradient_retention))
            or not 0.0 < minimum_complete_gradient_retention <= 1.0
        ):
            raise ValueError("minimum_complete_gradient_retention must lie in (0,1]")
        if campaign_config.validation_interval_updates != 512:
            raise ValueError("specification 1.61 validation cadence must be exactly 512 updates")
        if campaign_config.screen_maximum_updates > 512:
            raise ValueError("disposable screen may not exceed 512 updates")
        if (
            optimizer_config.warmup_updates != campaign_config.warmup_updates
            or optimizer_config.maximum_updates != campaign_config.maximum_updates
            or optimizer_config.final_learning_rate_fraction
            != campaign_config.final_learning_rate_fraction
        ):
            raise ValueError("optimizer and campaign schedules must be identical")

        self.model = model
        self.perception_owner = perception_owner
        self.relation_owner = relation_owner
        self.legacy_owners = tuple(legacy_owners)
        self.rows = _validate_training_rows(
            training_rows,
            screen_only=screen_only,
            test_only_synthetic_manifest=test_only_synthetic_manifest,
        )
        self.screen_only = screen_only
        self.test_only_synthetic_manifest = bool(test_only_synthetic_manifest)
        self.objective_adapter = objective_adapter
        self.resolved_config = _configuration_payload(resolved_config)
        self.source_provenance = deepcopy(dict(source_provenance))
        self.schedule_seed = schedule_seed
        self.materializer = materializer
        self.validation_hook = validation_hook
        self.runtime_reset = runtime_reset
        self.maximum_gradient_norm = float(maximum_gradient_norm)
        self.minimum_complete_gradient_retention = float(minimum_complete_gradient_retention)
        self.schedule = DynamicSetMicrobatchSchedule(self.rows, seed=schedule_seed)
        self.optimizer, self.scheduler, self.capacity = build_dynamic_set_optimizer(
            model,
            perception=perception_owner,
            relation=relation_owner,
            legacy_modules=self.legacy_owners,
            config=self.optimizer_config,
        )
        initial_optimizer_state = self.optimizer.state_dict()
        self._optimizer_group_static = tuple(
            {key: deepcopy(value) for key, value in group.items() if key not in {"lr", "params"}}
            for group in initial_optimizer_state["param_groups"]
        )
        initial_scheduler_state = self.scheduler.state_dict()
        self._scheduler_static = {
            key: deepcopy(value)
            for key, value in initial_scheduler_state.items()
            if key not in {"last_epoch", "_step_count", "_last_lr"}
        }
        self._owner_names = self._validate_owner_contract()
        self._validate_cpu_float32()
        self.bindings = make_dynamic_set_checkpoint_bindings(
            self.rows,
            resolved_config=self.resolved_config,
            source_provenance=self.source_provenance,
            optimizer_config=self.optimizer_config,
            campaign_config=self.campaign_config,
            loss_weights=self.loss_weights,
            maximum_gradient_norm=self.maximum_gradient_norm,
            minimum_complete_gradient_retention=self.minimum_complete_gradient_retention,
        )
        self._completed_updates = 0
        self._rejected_update_count = 0
        self._rejected_optimizer_mutation_count = 0
        self._minimum_complete_gradient_retention = 1.0
        self._integrity_poisoned = False
        self._validation_records: list[DynamicSetValidationRecord] = []
        self._last_update: DynamicSetUpdateReport | None = None
        self._assert_complete_state(0)

    @classmethod
    def from_online_world_model(
        cls,
        *,
        model: nn.Module,
        training_rows: Sequence[PhysicalManifestRow],
        objective_adapter: DynamicSetObjectiveAdapter,
        resolved_config: object,
        source_provenance: Mapping[str, Any],
        schedule_seed: int,
        materializer: Materializer = materialize_dynamic_set_episode,
        validation_hook: ValidationHook | None = None,
        optimizer_config: DynamicSetOptimizerConfig = DEFAULT_DYNAMIC_SET_OPTIMIZER_CONFIG,
        campaign_config: DynamicSetCampaignConfig = DEFAULT_CAMPAIGN,
        loss_weights: DynamicSetLossWeights = DEFAULT_DYNAMIC_SET_LOSS_WEIGHTS,
        maximum_gradient_norm: float = DEFAULT_MAXIMUM_GRADIENT_NORM,
        minimum_complete_gradient_retention: float = MINIMUM_COMPLETE_GRADIENT_RETENTION,
        screen_only: bool = False,
        test_only_synthetic_manifest: bool = False,
    ) -> DynamicSetTrainer:
        perception, relation, legacy = _resolve_online_world_model_owners(model, resolved_config)

        def reset_runtime(module: nn.Module, batch_size: int) -> None:
            assert hasattr(module, "reset")
            module.reset(batch_size=batch_size)

        return cls(
            model=model,
            perception_owner=perception,
            relation_owner=relation,
            legacy_owners=legacy,
            training_rows=training_rows,
            objective_adapter=objective_adapter,
            resolved_config=resolved_config,
            source_provenance=source_provenance,
            schedule_seed=schedule_seed,
            materializer=materializer,
            validation_hook=validation_hook,
            runtime_reset=reset_runtime,
            optimizer_config=optimizer_config,
            campaign_config=campaign_config,
            loss_weights=loss_weights,
            maximum_gradient_norm=maximum_gradient_norm,
            minimum_complete_gradient_retention=minimum_complete_gradient_retention,
            screen_only=screen_only,
            test_only_synthetic_manifest=test_only_synthetic_manifest,
        )

    @property
    def completed_updates(self) -> int:
        return self._completed_updates

    @property
    def next_update_index(self) -> int:
        return self._completed_updates

    @property
    def rejected_update_count(self) -> int:
        return self._rejected_update_count

    @property
    def rejected_optimizer_mutation_count(self) -> int:
        return self._rejected_optimizer_mutation_count

    @property
    def observed_minimum_complete_gradient_retention(self) -> float:
        return self._minimum_complete_gradient_retention

    @property
    def validation_records(self) -> tuple[DynamicSetValidationRecord, ...]:
        return tuple(self._validation_records)

    @property
    def last_update(self) -> DynamicSetUpdateReport | None:
        return self._last_update

    def _validate_cpu_float32(self) -> None:
        tensors = (*self.model.parameters(), *self.model.buffers())
        for value in tensors:
            if value.device.type != "cpu":
                raise ValueError("specification-1.61 optimization is CPU-only")
            if value.is_floating_point() and value.dtype is not torch.float32:
                raise ValueError("specification-1.61 optimization requires float32 tensors")

    def _validate_owner_contract(self) -> Mapping[str, tuple[str, ...]]:
        complete = {id(parameter): name for name, parameter in self.model.named_parameters()}
        perception_ids = {id(parameter) for parameter in self.perception_owner.parameters()}
        relation_ids = {id(parameter) for parameter in self.relation_owner.parameters()}
        legacy_parameters = tuple(
            parameter for module in self.legacy_owners for parameter in module.parameters()
        )
        legacy_ids = {id(parameter) for parameter in legacy_parameters}
        if len(legacy_ids) != len(legacy_parameters):
            raise ValueError("legacy parameter owners overlap")
        if (
            not perception_ids
            or not relation_ids
            or perception_ids & relation_ids
            or perception_ids & legacy_ids
            or relation_ids & legacy_ids
        ):
            raise ValueError("optimizer and legacy owners must be nonempty and disjoint")
        if perception_ids | relation_ids | legacy_ids != set(complete):
            raise ValueError("optimizer/legacy owners must exhaust the complete model")
        optimizer_ids = {
            id(parameter) for group in self.optimizer.param_groups for parameter in group["params"]
        }
        if optimizer_ids != perception_ids | relation_ids:
            raise ValueError("AdamW parameters differ from the two declared owners")
        if [group.get("name") for group in self.optimizer.param_groups] != [
            "perception",
            "relation",
        ]:
            raise ValueError("AdamW must contain exact perception and relation groups")
        if any(
            parameter.requires_grad != (id(parameter) in optimizer_ids)
            for parameter in self.model.parameters()
        ):
            raise ValueError("requires_grad differs from exact AdamW ownership")
        return {
            "perception": tuple(sorted(complete[identity] for identity in perception_ids)),
            "relation": tuple(sorted(complete[identity] for identity in relation_ids)),
            "legacy": tuple(sorted(complete[identity] for identity in legacy_ids)),
        }

    def _owner_parameters(self, owner: str | None = None) -> tuple[nn.Parameter, ...]:
        groups = self.optimizer.param_groups
        if owner is None:
            return tuple(parameter for group in groups for parameter in group["params"])
        return tuple(
            parameter
            for group in groups
            if group.get("name") == owner
            for parameter in group["params"]
        )

    def _reset(self, batch_size: int) -> None:
        if self.runtime_reset is not None:
            self.runtime_reset(self.model, batch_size)

    def _snapshot_transaction(self) -> dict[str, Any]:
        return {
            "model": _clone_model_state(self.model),
            "optimizer": deepcopy(self.optimizer.state_dict()),
            "scheduler": deepcopy(self.scheduler.state_dict()),
            "adapter": deepcopy(dict(self.objective_adapter.state_dict())),
            "rng": _capture_rng_state(),
            "mode": self.model.training,
        }

    def _restore_transaction(self, snapshot: Mapping[str, Any]) -> None:
        self.model.load_state_dict(snapshot["model"], strict=True)
        self.optimizer.load_state_dict(snapshot["optimizer"])
        self.scheduler.load_state_dict(snapshot["scheduler"])
        self.objective_adapter.load_state_dict(deepcopy(snapshot["adapter"]))
        self.model.train(bool(snapshot["mode"]))
        self._reset(1)
        _restore_rng_state(snapshot["rng"])
        self.optimizer.zero_grad(set_to_none=True)

    def _reject(
        self, snapshot: Mapping[str, Any], error: BaseException
    ) -> DynamicSetUpdateRejected:
        try:
            self._restore_transaction(snapshot)
            restored = self._snapshot_transaction()
            if not _trees_equal(restored, snapshot):
                self._rejected_optimizer_mutation_count += 1
                self._integrity_poisoned = True
                raise RuntimeError("dynamic-set rejected update did not restore byte-exact state")
        except BaseException as rollback_error:
            self._rejected_optimizer_mutation_count += int(not self._integrity_poisoned)
            self._integrity_poisoned = True
            raise RuntimeError("dynamic-set update rollback failed") from rollback_error
        self._rejected_update_count += 1
        reason = f"{type(error).__name__}: {error}"
        return DynamicSetUpdateRejected(reason)

    def _gradient_norm(self, parameters: Sequence[nn.Parameter]) -> float:
        square_sum = math.fsum(
            float(parameter.grad.detach().to(torch.float64).square().sum())
            for parameter in parameters
            if parameter.grad is not None
        )
        return math.sqrt(square_sum)

    def _validate_and_clip_gradients(self) -> tuple[float, float, float, float, float]:
        owners = self._owner_parameters()
        for parameter in owners:
            gradient = parameter.grad
            if gradient is None:
                raise FloatingPointError("every AdamW-owned parameter must receive a gradient")
            if gradient.is_sparse:
                raise FloatingPointError("sparse owner gradients are unsupported")
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError("AdamW owner gradient contains NaN or Inf")
        perception_norm = self._gradient_norm(self._owner_parameters("perception"))
        relation_norm = self._gradient_norm(self._owner_parameters("relation"))
        raw_norm = self._gradient_norm(owners)
        if not math.isfinite(raw_norm):
            raise FloatingPointError("complete owner gradient norm is nonfinite")
        coefficient = min(1.0, self.maximum_gradient_norm / max(raw_norm, 1.0e-30))
        if coefficient < self.minimum_complete_gradient_retention:
            raise FloatingPointError(
                "complete-gradient retention "
                f"{coefficient:.12g} is below {self.minimum_complete_gradient_retention:.12g}"
            )
        if coefficient < 1.0:
            for parameter in owners:
                assert parameter.grad is not None
                parameter.grad.mul_(coefficient)
        applied_norm = raw_norm * coefficient
        return perception_norm, relation_norm, raw_norm, applied_norm, coefficient

    def _assert_optimizer_alignment(self, completed_updates: int) -> None:
        state = self.optimizer.state_dict()
        _assert_finite_tree(state, root="optimizer_state")
        self._validate_serialized_optimizer_state(state, completed_updates)

    def _expected_learning_rates(self, completed_updates: int) -> tuple[float, float]:
        multiplier = optimizer_learning_rate_multiplier(
            completed_updates,
            self.optimizer_config,
        )
        return (
            self.optimizer_config.perception_learning_rate * multiplier,
            self.optimizer_config.relation_learning_rate * multiplier,
        )

    def _validate_scheduler_state(
        self,
        state: Mapping[str, Any],
        completed_updates: int,
    ) -> None:
        required = {*self._scheduler_static, "last_epoch", "_step_count", "_last_lr"}
        if set(state) != required:
            raise ValueError("scheduler state schema differs")
        for name, expected in self._scheduler_static.items():
            if state[name] != expected:
                raise ValueError(f"scheduler static field {name!r} differs")
        if state["last_epoch"] != completed_updates:
            raise ValueError("scheduler is not aligned with completed optimizer updates")
        if state["_step_count"] != completed_updates + 1:
            raise ValueError("scheduler call count is not exact")
        expected_rates = list(self._expected_learning_rates(completed_updates))
        if state["_last_lr"] != expected_rates:
            raise ValueError("scheduler learning rates are not exact")

    def _assert_complete_state(self, completed_updates: int) -> None:
        _assert_finite_tree(self.model.state_dict(), root="model_state")
        self._assert_optimizer_alignment(completed_updates)
        scheduler_state = self.scheduler.state_dict()
        _assert_finite_tree(scheduler_state, root="scheduler_state")
        self._validate_scheduler_state(scheduler_state, completed_updates)
        adapter_state = self.objective_adapter.state_dict()
        if not isinstance(adapter_state, Mapping):
            raise TypeError("objective adapter state_dict must return a mapping")
        _assert_finite_tree(adapter_state, root="adapter_state")

    def _materialize_microbatch(
        self,
        update_index: int,
        microbatch_index: int,
        dataset_indices: tuple[int, ...],
    ) -> DynamicSetTrainingMicrobatch:
        if len(dataset_indices) != MICROBATCH_SIZE:
            raise ValueError("dynamic-set schedule emitted a non-B4 microbatch")
        rows = tuple(self.rows[index] for index in dataset_indices)
        contextual = getattr(self.materializer, "materialize_for_training", None)
        if callable(contextual):
            perception_frame_index = dynamic_set_perception_frame_index(
                update_index,
                microbatch_index,
            )
            materializations = tuple(
                contextual(
                    row,
                    perception_frame_index=perception_frame_index,
                )
                for row in rows
            )
        else:
            materializations = tuple(self.materializer(row) for row in rows)
        return DynamicSetTrainingMicrobatch(
            update_index=update_index,
            microbatch_index=microbatch_index,
            dataset_indices=dataset_indices,
            rows=rows,
            materializations=materializations,
        )

    def _read_only_metrics(
        self,
        callback: Callable[[], Mapping[str, Any] | DynamicSetScreenSnapshot],
    ) -> Mapping[str, Any] | DynamicSetScreenSnapshot:
        model_state = _clone_model_state(self.model)
        rng_state = _capture_rng_state()
        mode = self.model.training
        try:
            self.model.eval()
            with torch.inference_mode():
                result = callback()
            _assert_state_equal(self.model.state_dict(), model_state)
            return result
        finally:
            self.model.load_state_dict(model_state, strict=True)
            self.model.train(mode)
            self._reset(1)
            _restore_rng_state(rng_state)

    def _validation_if_due(self) -> DynamicSetValidationRecord | None:
        if self._completed_updates % self.campaign_config.validation_interval_updates:
            return None
        if self.validation_hook is None:
            return None
        raw = self._read_only_metrics(
            lambda: self.validation_hook(self.model, self._completed_updates)
        )
        if not isinstance(raw, Mapping):
            raise TypeError("validation hook must return a finite scalar mapping")
        record = DynamicSetValidationRecord(
            completed_updates=self._completed_updates,
            metrics=_numeric_metrics(raw, label="validation_metrics"),
        )
        self._validation_records.append(record)
        return record

    def run_update(self) -> DynamicSetUpdateReport:
        """Apply one exact six-B4 update or reject it before lasting mutation."""

        if (
            self.screen_only
            and self._completed_updates >= self.campaign_config.screen_maximum_updates
        ):
            raise RuntimeError("disposable-screen trainer has reached its 512-update hard cap")
        if self._completed_updates >= self.campaign_config.maximum_updates:
            raise RuntimeError("dynamic-set campaign has reached its hard update cap")
        snapshot = self._snapshot_transaction()
        update_index = self._completed_updates
        term_totals = {item.name: 0.0 for item in fields(DynamicSetLosses) if item.name != "total"}
        objective_total = 0.0
        cell_counts = [0] * PHYSICAL_CELL_COUNT
        try:
            self._assert_complete_state(update_index)
            self.model.train(True)
            self.optimizer.zero_grad(set_to_none=True)
            groups = self.schedule.microbatches_for_update(update_index)
            if len(groups) != MICROBATCHES_PER_UPDATE:
                raise ValueError("dynamic-set schedule must emit exactly six microbatches")
            for microbatch_index, dataset_indices in enumerate(groups):
                microbatch = self._materialize_microbatch(
                    update_index,
                    microbatch_index,
                    dataset_indices,
                )
                for row in microbatch.rows:
                    cell_counts[row.cell_index] += 1
                self._reset(MICROBATCH_SIZE)
                objective_inputs = self.objective_adapter.build_objective_inputs(
                    self.model,
                    microbatch,
                )
                _validate_objective_inputs(objective_inputs, microbatch)
                losses = dynamic_set_objective(
                    objective_inputs.perception,
                    objective_inputs.dynamics,
                    weights=self.loss_weights,
                )
                for item in fields(losses):
                    value = getattr(losses, item.name)
                    if (
                        not isinstance(value, Tensor)
                        or value.shape != ()
                        or not bool(torch.isfinite(value))
                    ):
                        raise FloatingPointError(
                            f"physical objective {item.name!r} must be one finite scalar"
                        )
                if not losses.total.requires_grad:
                    raise FloatingPointError("physical objective is disconnected from AdamW owners")
                objective_total += float(losses.total.detach())
                for name in term_totals:
                    term_totals[name] += float(getattr(losses, name).detach())
                (losses.total / MICROBATCHES_PER_UPDATE).backward()
                self._reset(1)

            if set(index for index, count in enumerate(cell_counts) if count) != set(
                range(PHYSICAL_CELL_COUNT)
            ):
                raise ValueError("optimizer update does not span all 22 physical cells")
            if sum(cell_counts) != EXAMPLES_PER_UPDATE or sorted(cell_counts).count(2) != 2:
                raise ValueError(
                    "optimizer update must contain 22 cells plus two rotating duplicates"
                )
            # Forward/adapter code may not mutate a parameter or buffer before
            # the audited AdamW boundary.
            _assert_state_equal(self.model.state_dict(), snapshot["model"])
            self._assert_complete_state(update_index)
            (
                perception_norm,
                relation_norm,
                raw_norm,
                applied_norm,
                retention,
            ) = self._validate_and_clip_gradients()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            next_completed = update_index + 1
            self._assert_complete_state(next_completed)
            owner_names = frozenset(
                (*self._owner_names["perception"], *self._owner_names["relation"])
            )
            _assert_state_equal(
                self.model.state_dict(),
                snapshot["model"],
                excluded=owner_names,
            )
        except BaseException as error:
            raise self._reject(snapshot, error) from error

        self._completed_updates += 1
        self._minimum_complete_gradient_retention = min(
            self._minimum_complete_gradient_retention,
            retention,
        )
        validation = self._validation_if_due()
        report = DynamicSetUpdateReport(
            completed_updates=self._completed_updates,
            next_update_index=self._completed_updates,
            materialized_examples=EXAMPLES_PER_UPDATE,
            cell_counts=tuple(cell_counts),
            objective=objective_total / MICROBATCHES_PER_UPDATE,
            loss_terms={
                name: value / MICROBATCHES_PER_UPDATE for name, value in term_totals.items()
            },
            perception_gradient_norm=perception_norm,
            relation_gradient_norm=relation_norm,
            raw_gradient_norm=raw_norm,
            applied_gradient_norm=applied_norm,
            complete_gradient_retention=retention,
            finite_owner_gradients=True,
            validation=validation,
        )
        self._last_update = report
        return report

    def run_updates(self, count: int) -> tuple[DynamicSetUpdateReport, ...]:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("update count must be a nonnegative integer")
        if self._completed_updates + count > self.campaign_config.maximum_updates:
            raise ValueError("requested updates exceed the campaign hard cap")
        return tuple(self.run_update() for _ in range(count))

    def _screen_snapshot(self, hook: ScreenHook) -> DynamicSetScreenSnapshot:
        raw = self._read_only_metrics(lambda: hook(self.model, self._completed_updates))
        if not isinstance(raw, DynamicSetScreenSnapshot):
            raise TypeError("screen hook must return DynamicSetScreenSnapshot")
        if raw.example_count != SCREEN_EXAMPLE_COUNT:
            raise ValueError("disposable screen must evaluate exactly 64 examples")
        for name in ("optimization_objective", "proposal_f1", "collision_f1"):
            value = getattr(raw, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"screen {name} must be finite")
        return raw

    def run_disposable_screen(
        self,
        *,
        updates: int,
        evaluation_hook: ScreenHook,
    ) -> DynamicSetScreenReport:
        """Run a fresh, exact-64-example screen for at most 512 updates."""

        if not self.screen_only:
            raise RuntimeError(
                "a disposable screen requires a dedicated exact-64-row screen trainer"
            )
        if self.validation_hook is not None:
            raise RuntimeError("a disposable screen cannot install a development validation hook")
        if self._completed_updates != 0 or self._rejected_update_count != 0:
            raise RuntimeError("a disposable screen must start from a fresh trainer")
        if (
            isinstance(updates, bool)
            or not isinstance(updates, int)
            or not 0 < updates <= self.campaign_config.screen_maximum_updates
        ):
            raise ValueError("disposable screen updates must lie in [1,512]")
        initial = self._screen_snapshot(evaluation_hook)
        reports: list[DynamicSetUpdateReport] = []
        try:
            for _ in range(updates):
                reports.append(self.run_update())
        except DynamicSetUpdateRejected:
            pass
        final = self._screen_snapshot(evaluation_hook)
        metric_values: dict[str, Any] = {
            "initial_optimization_objective": initial.optimization_objective,
            "final_optimization_objective": final.optimization_objective,
            "initial_objective_regret": initial.objective_regret,
            "final_objective_regret": final.objective_regret,
            "proposal_f1": final.proposal_f1,
            "collision_f1": final.collision_f1,
            "finite_owner_gradients": all(report.finite_owner_gradients for report in reports),
            "rejected_update_count": self._rejected_update_count,
            "completed_updates": len(reports),
        }
        # The campaign dataclass may grow the now-explicit example count; keep
        # the engine compatible while still enforcing it independently here.
        if "example_count" in {item.name for item in fields(DisposableScreenMetrics)}:
            metric_values["example_count"] = SCREEN_EXAMPLE_COUNT
        metrics = DisposableScreenMetrics(**metric_values)
        failures = list(disposable_screen_failures(metrics))
        if (
            initial.example_count != SCREEN_EXAMPLE_COUNT
            or final.example_count != SCREEN_EXAMPLE_COUNT
        ):
            failures.append("example_count:!=64")
        return DynamicSetScreenReport(
            metrics=metrics,
            example_count=SCREEN_EXAMPLE_COUNT,
            failures=tuple(failures),
        )

    def _next_sample_state(self, update_index: int | None = None) -> dict[str, Any]:
        index = self._completed_updates if update_index is None else update_index
        groups = self.schedule.microbatches_for_update(index)
        return {
            "absolute_update_index": index,
            "absolute_example_draw": index * EXAMPLES_PER_UPDATE,
            "schedule_seed": self.schedule_seed,
            "microbatch_dataset_indices": [list(group) for group in groups],
        }

    def checkpoint_payload(self) -> dict[str, Any]:
        """Build a complete, finite, exact-resume in-memory checkpoint."""

        if self._integrity_poisoned:
            raise RuntimeError("an integrity-poisoned trainer cannot emit a checkpoint")
        self._assert_complete_state(self._completed_updates)
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "bindings": asdict(self.bindings),
            "source_provenance": deepcopy(self.source_provenance),
            "screen_only": self.screen_only,
            "test_only_synthetic_manifest": self.test_only_synthetic_manifest,
            "manifest_row_count": len(self.rows),
            "owner_parameter_names": {
                name: list(values) for name, values in self._owner_names.items()
            },
            "model_state": _clone_model_state(self.model),
            "optimizer_state": deepcopy(self.optimizer.state_dict()),
            "scheduler_state": deepcopy(self.scheduler.state_dict()),
            "adapter_state": deepcopy(dict(self.objective_adapter.state_dict())),
            "trainer_state": {
                "completed_updates": self._completed_updates,
                "rejected_update_count": self._rejected_update_count,
                "rejected_optimizer_mutation_count": (self._rejected_optimizer_mutation_count),
                "minimum_complete_gradient_retention": (self._minimum_complete_gradient_retention),
                "validation_records": [
                    {
                        "completed_updates": record.completed_updates,
                        "metrics": dict(record.metrics),
                    }
                    for record in self._validation_records
                ],
            },
            "next_sample_state": self._next_sample_state(),
            "rng_state": _capture_rng_state(),
        }
        payload["model_state_sha256"] = dynamic_set_model_state_sha256(payload["model_state"])
        for key in (
            "model_state",
            "optimizer_state",
            "scheduler_state",
            "adapter_state",
        ):
            _assert_finite_tree(payload[key], root=key)
        return payload

    def _validate_checkpoint_payload(
        self, payload: Mapping[str, Any]
    ) -> tuple[int, int, int, float, list[DynamicSetValidationRecord]]:
        required = {
            "schema",
            "bindings",
            "source_provenance",
            "screen_only",
            "test_only_synthetic_manifest",
            "manifest_row_count",
            "owner_parameter_names",
            "model_state",
            "model_state_sha256",
            "optimizer_state",
            "scheduler_state",
            "adapter_state",
            "trainer_state",
            "next_sample_state",
            "rng_state",
        }
        if set(payload) != required or payload.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("dynamic-set checkpoint schema is incompatible")
        if payload["bindings"] != asdict(self.bindings):
            raise ValueError("checkpoint manifest/protocol/config/source binding differs")
        if canonical_sha256(payload["source_provenance"]) != self.bindings.source_sha256:
            raise ValueError("checkpoint source provenance differs from its binding")
        if payload["screen_only"] is not self.screen_only:
            raise ValueError("checkpoint screen-only authority differs")
        if payload["test_only_synthetic_manifest"] is not self.test_only_synthetic_manifest:
            raise ValueError("checkpoint synthetic-manifest authority differs")
        if payload["manifest_row_count"] != len(self.rows):
            raise ValueError("checkpoint manifest row count differs")
        expected_owner_names = {name: list(values) for name, values in self._owner_names.items()}
        if payload["owner_parameter_names"] != expected_owner_names:
            raise ValueError("checkpoint AdamW owner names differ")

        trainer_state = payload["trainer_state"]
        if not isinstance(trainer_state, Mapping) or set(trainer_state) != {
            "completed_updates",
            "rejected_update_count",
            "rejected_optimizer_mutation_count",
            "minimum_complete_gradient_retention",
            "validation_records",
        }:
            raise ValueError("checkpoint trainer state is incompatible")
        completed = trainer_state["completed_updates"]
        rejected = trainer_state["rejected_update_count"]
        rejected_mutations = trainer_state["rejected_optimizer_mutation_count"]
        minimum_retention = trainer_state["minimum_complete_gradient_retention"]
        if (
            isinstance(completed, bool)
            or not isinstance(completed, int)
            or not 0
            <= completed
            <= (
                self.campaign_config.screen_maximum_updates
                if self.screen_only
                else self.campaign_config.maximum_updates
            )
        ):
            raise ValueError("checkpoint completed_updates is invalid")
        if isinstance(rejected, bool) or not isinstance(rejected, int) or rejected < 0:
            raise ValueError("checkpoint rejected_update_count is invalid")
        if (
            isinstance(rejected_mutations, bool)
            or not isinstance(rejected_mutations, int)
            or rejected_mutations < 0
        ):
            raise ValueError("checkpoint rejected_optimizer_mutation_count is invalid")
        if (
            isinstance(minimum_retention, bool)
            or not isinstance(minimum_retention, Real)
            or not math.isfinite(float(minimum_retention))
            or not self.minimum_complete_gradient_retention <= float(minimum_retention) <= 1.0
            or (completed == 0 and float(minimum_retention) != 1.0)
        ):
            raise ValueError("checkpoint minimum complete-gradient retention is invalid")
        if payload["next_sample_state"] != self._next_sample_state(completed):
            raise ValueError("checkpoint next sample state is not exact")

        raw_records = trainer_state["validation_records"]
        if not isinstance(raw_records, list):
            raise TypeError("checkpoint validation records must be a list")
        records: list[DynamicSetValidationRecord] = []
        previous = 0
        for raw in raw_records:
            if not isinstance(raw, Mapping) or set(raw) != {"completed_updates", "metrics"}:
                raise ValueError("checkpoint validation record is malformed")
            step = raw["completed_updates"]
            if (
                isinstance(step, bool)
                or not isinstance(step, int)
                or step <= previous
                or step > completed
                or step % self.campaign_config.validation_interval_updates
            ):
                raise ValueError("checkpoint validation cadence is invalid")
            records.append(
                DynamicSetValidationRecord(
                    completed_updates=step,
                    metrics=_numeric_metrics(raw["metrics"], label="validation_metrics"),
                )
            )
            previous = step

        model_state = payload["model_state"]
        current_state = self.model.state_dict()
        if not isinstance(model_state, Mapping) or set(model_state) != set(current_state):
            raise ValueError("checkpoint model state schema differs")
        for name, expected in current_state.items():
            value = model_state[name]
            if (
                not isinstance(value, Tensor)
                or value.shape != expected.shape
                or value.dtype != expected.dtype
            ):
                raise ValueError(f"checkpoint model tensor {name!r} is incompatible")
        if validated_sha256(
            payload["model_state_sha256"], label="checkpoint model state"
        ) != dynamic_set_model_state_sha256(model_state):
            raise ValueError("checkpoint model-state digest differs")
        for key in ("model_state", "optimizer_state", "scheduler_state", "adapter_state"):
            _assert_finite_tree(payload[key], root=key)
        scheduler_state = payload["scheduler_state"]
        if not isinstance(scheduler_state, Mapping):
            raise TypeError("checkpoint scheduler state must be a mapping")
        self._validate_scheduler_state(scheduler_state, completed)
        optimizer_state = payload["optimizer_state"]
        if not isinstance(optimizer_state, Mapping):
            raise TypeError("checkpoint optimizer state must be a mapping")
        self._validate_serialized_optimizer_state(optimizer_state, completed)
        if not isinstance(payload["adapter_state"], Mapping):
            raise TypeError("checkpoint adapter state must be a mapping")
        if not isinstance(payload["rng_state"], Mapping):
            raise TypeError("checkpoint RNG state must be a mapping")
        return completed, rejected, rejected_mutations, float(minimum_retention), records

    def _validate_serialized_optimizer_state(
        self,
        state: Mapping[str, Any],
        completed_updates: int,
    ) -> None:
        if set(state) != {"state", "param_groups"}:
            raise ValueError("checkpoint optimizer state schema differs")
        groups = state["param_groups"]
        values = state["state"]
        if not isinstance(groups, list) or not isinstance(values, Mapping) or len(groups) != 2:
            raise ValueError("checkpoint optimizer must contain exactly two groups")
        expected_rates = self._expected_learning_rates(completed_updates)
        parameter_ids: list[int] = []
        parameter_pairs: list[tuple[int, nn.Parameter]] = []
        for index, (group, static, live_group, expected_rate) in enumerate(
            zip(
                groups,
                self._optimizer_group_static,
                self.optimizer.param_groups,
                expected_rates,
                strict=True,
            )
        ):
            if not isinstance(group, Mapping):
                raise TypeError("checkpoint optimizer groups must be mappings")
            if set(group) != {*static, "lr", "params"}:
                raise ValueError("checkpoint optimizer group schema differs")
            for name, expected in static.items():
                if group[name] != expected:
                    raise ValueError(f"checkpoint optimizer group {index} field {name!r} differs")
            rate = group["lr"]
            if (
                isinstance(rate, bool)
                or not isinstance(rate, Real)
                or not math.isfinite(float(rate))
                or float(rate) != expected_rate
            ):
                raise ValueError("checkpoint optimizer learning rate is not exact")
            serialized = group["params"]
            live = live_group["params"]
            if (
                not isinstance(serialized, list)
                or len(serialized) != len(live)
                or any(type(parameter_id) is not int for parameter_id in serialized)
            ):
                raise ValueError("checkpoint optimizer parameter layout differs")
            parameter_ids.extend(serialized)
            parameter_pairs.extend(zip(serialized, live, strict=True))
        if len(set(parameter_ids)) != len(parameter_ids):
            raise ValueError("checkpoint optimizer parameter identifiers overlap")
        if completed_updates == 0:
            if values:
                raise ValueError("checkpoint step-zero optimizer state must be empty")
            return
        if set(values) != set(parameter_ids):
            raise ValueError("checkpoint optimizer state is incomplete")
        for parameter_id, parameter in parameter_pairs:
            parameter_state = values[parameter_id]
            if not isinstance(parameter_state, Mapping) or set(parameter_state) != {
                "step",
                "exp_avg",
                "exp_avg_sq",
            }:
                raise ValueError("checkpoint Adam parameter state schema differs")
            step = parameter_state["step"]
            if (
                not isinstance(step, Tensor)
                or step.shape != ()
                or step.dtype is not torch.float32
                or step.device.type != "cpu"
                or float(step) != float(completed_updates)
            ):
                raise ValueError("checkpoint Adam step is not exact")
            for name in ("exp_avg", "exp_avg_sq"):
                moment = parameter_state[name]
                if (
                    not isinstance(moment, Tensor)
                    or moment.shape != parameter.shape
                    or moment.dtype != parameter.dtype
                    or moment.device != parameter.device
                ):
                    raise ValueError(f"checkpoint Adam {name} layout differs")

    def load_checkpoint_payload(
        self,
        payload: Mapping[str, Any],
        *,
        restore_rng: bool = True,
    ) -> None:
        """Atomically restore an exact continuation from a trusted payload."""

        if not isinstance(payload, Mapping):
            raise TypeError("dynamic-set checkpoint payload must be a mapping")
        completed, rejected, rejected_mutations, minimum_retention, records = (
            self._validate_checkpoint_payload(payload)
        )
        snapshot = self._snapshot_transaction()
        old_completed = self._completed_updates
        old_rejected = self._rejected_update_count
        old_rejected_mutations = self._rejected_optimizer_mutation_count
        old_minimum_retention = self._minimum_complete_gradient_retention
        old_integrity_poisoned = self._integrity_poisoned
        old_records = list(self._validation_records)
        old_last = self._last_update
        try:
            self.model.load_state_dict(payload["model_state"], strict=True)
            self.optimizer.load_state_dict(deepcopy(payload["optimizer_state"]))
            self.scheduler.load_state_dict(deepcopy(payload["scheduler_state"]))
            self.objective_adapter.load_state_dict(deepcopy(payload["adapter_state"]))
            self._completed_updates = completed
            self._rejected_update_count = rejected
            self._rejected_optimizer_mutation_count = rejected_mutations
            self._minimum_complete_gradient_retention = minimum_retention
            self._integrity_poisoned = bool(rejected_mutations)
            self._validation_records = records
            self._last_update = None
            self._assert_complete_state(completed)
            self._validate_cpu_float32()
            self._reset(1)
            if restore_rng:
                _restore_rng_state(payload["rng_state"])
        except BaseException:
            self._restore_transaction(snapshot)
            self._completed_updates = old_completed
            self._rejected_update_count = old_rejected
            self._rejected_optimizer_mutation_count = old_rejected_mutations
            self._minimum_complete_gradient_retention = old_minimum_retention
            self._integrity_poisoned = old_integrity_poisoned
            self._validation_records = old_records
            self._last_update = old_last
            raise


__all__ = [
    "CHECKPOINT_SCHEMA",
    "DEFAULT_MAXIMUM_GRADIENT_NORM",
    "MINIMUM_COMPLETE_GRADIENT_RETENTION",
    "SCREEN_EXAMPLE_COUNT",
    "SCREEN_MANIFEST_SHA256",
    "SCREEN_ROWS",
    "DynamicSetCausalSupport",
    "DynamicSetCheckpointBindings",
    "DynamicSetObjectiveAdapter",
    "DynamicSetObjectiveInputs",
    "DynamicSetScreenReport",
    "DynamicSetScreenSnapshot",
    "DynamicSetTrainer",
    "DynamicSetTrainingMicrobatch",
    "DynamicSetUpdateRejected",
    "DynamicSetUpdateReport",
    "DynamicSetValidationRecord",
    "causal_scene_predictable_mask",
    "dynamic_set_training_protocol_sha256",
    "dynamic_set_model_state_sha256",
    "dynamic_set_perception_frame_index",
    "make_dynamic_set_checkpoint_bindings",
]
