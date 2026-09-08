"""Campaign controls for the specification-1.61 CPU qualification."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Literal

from world_model.training.convergence import CampaignInspection, decide_continuation
from world_model.training.dynamic_set_objectives import dynamic_set_objective_regret
from world_model.training.qualification_core import canonical_sha256, validated_sha256

CampaignStatus = Literal[
    "continue",
    "qualified_convergence",
    "objective_plateau",
    "failed_to_improve",
    "limit_hit",
]
SecondAttempt = Literal["none", "widen_perception", "widen_relation"]
LimitHitReason = Literal[
    "none",
    "training_reserve_boundary",
    "minimum_update_projection_infeasible",
]

# Every production cell owns exactly 3,000 rows and is drawn at least once per
# optimizer update.  Thus update indices 0--2,999 complete the deterministic
# first-visit cache fill.  The next 512-aligned boundary is 3,072; updates
# 3,072--3,583 are the first complete validation-aligned window containing
# only warm-cache updates.  Earlier update rates remain charged as sunk wall
# time, but may not be extrapolated over the remainder of the campaign.
TRAINING_CACHE_FULL_COVERAGE_UPDATES = 3_000
TRAINING_CACHE_WARM_WINDOW_START_UPDATES = 3_072
TRAINING_CACHE_TIMING_SUPPORT_UPDATES = 3_584
_SECOND_ATTEMPT_ADMISSION_SCHEMA = "dynamic_set_second_attempt_admission_v1"
_EXECUTION_TIMING_EVIDENCE_SCHEMA = "dynamic_set_update_timing_v3"
_ZERO_SHA256 = "0" * 64


@dataclass(frozen=True)
class DynamicSetCampaignConfig:
    minimum_updates: int = 16_384
    extension_updates: int = 4_096
    maximum_updates: int = 32_768
    validation_interval_updates: int = 512
    plateau_validation_count: int = 4
    minimum_relative_gain: float = 0.01
    warmup_updates: int = 512
    final_learning_rate_fraction: float = 0.10
    maximum_training_hours: float = 60.0
    reserved_audit_hours: float = 12.0
    minimum_timing_support_updates: int = TRAINING_CACHE_TIMING_SUPPORT_UPDATES
    timing_projection_window_updates: int = 512
    timing_projection_confidence_z: float = 2.576
    timing_projection_safety_factor: float = 1.10
    screen_maximum_updates: int = 512

    def validate(self) -> DynamicSetCampaignConfig:
        integer_values = (
            self.minimum_updates,
            self.extension_updates,
            self.maximum_updates,
            self.validation_interval_updates,
            self.plateau_validation_count,
            self.warmup_updates,
            self.minimum_timing_support_updates,
            self.timing_projection_window_updates,
            self.screen_maximum_updates,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in integer_values
        ):
            raise ValueError("campaign update/count controls must be positive integers")
        if self.maximum_updates < self.minimum_updates:
            raise ValueError("maximum_updates cannot be smaller than minimum_updates")
        if self.extension_updates % self.validation_interval_updates:
            raise ValueError("extensions must end on a complete validation boundary")
        if self.minimum_updates % self.validation_interval_updates:
            raise ValueError("minimum_updates must end on a complete validation boundary")
        if (self.maximum_updates - self.minimum_updates) % self.extension_updates:
            raise ValueError("maximum_updates must end on a complete extension boundary")
        if not 0.0 < self.minimum_relative_gain < 1.0:
            raise ValueError("minimum_relative_gain must lie in (0,1)")
        if not 0.0 < self.final_learning_rate_fraction <= 1.0:
            raise ValueError("final_learning_rate_fraction must lie in (0,1]")
        if (
            not math.isfinite(self.maximum_training_hours)
            or not math.isfinite(self.reserved_audit_hours)
            or self.maximum_training_hours <= 0.0
            or self.reserved_audit_hours <= 0.0
            or self.reserved_audit_hours >= self.maximum_training_hours
        ):
            raise ValueError("campaign wall-time controls must be positive")
        if self.minimum_timing_support_updates > self.minimum_updates:
            raise ValueError("timing support cannot exceed the minimum campaign")
        if self.timing_projection_window_updates > self.minimum_timing_support_updates:
            raise ValueError("timing projection support cannot be smaller than its window")
        if self.timing_projection_window_updates > self.maximum_updates:
            raise ValueError("timing projection window cannot exceed the campaign")
        if (
            not math.isfinite(self.timing_projection_confidence_z)
            or self.timing_projection_confidence_z <= 0.0
            or not math.isfinite(self.timing_projection_safety_factor)
            or self.timing_projection_safety_factor < 1.0
        ):
            raise ValueError("timing projection controls must be finite and conservative")
        return self

    @property
    def training_mutation_hours(self) -> float:
        """Wall time available to optimizer mutation after the audit reserve."""

        return self.maximum_training_hours - self.reserved_audit_hours

    @property
    def training_mutation_seconds(self) -> float:
        return self.training_mutation_hours * 3600.0

    @property
    def reserved_audit_seconds(self) -> float:
        return self.reserved_audit_hours * 3600.0


DEFAULT_CAMPAIGN = DynamicSetCampaignConfig().validate()


@dataclass(frozen=True)
class DisposableScreenMetrics:
    example_count: int
    initial_optimization_objective: float
    final_optimization_objective: float
    initial_objective_regret: float
    final_objective_regret: float
    proposal_f1: float
    collision_f1: float
    finite_owner_gradients: bool
    rejected_update_count: int
    completed_updates: int


@dataclass(frozen=True)
class DynamicSetCampaignDecision:
    status: CampaignStatus
    reason: str
    next_total_updates: int | None


@dataclass(frozen=True, slots=True)
class DynamicSetTimeProjection:
    """Deterministic feasibility result from durable completed-update timings.

    Projection is intentionally unavailable until the finite first-visit
    training-cache fill has completed and a subsequent complete warm support
    window is measured.  Once available, the rate is the larger of a 99%
    upper confidence mean and the empirical p90, with an additional frozen
    safety factor.  Every cold update and sunk rejected/rolled-back attempt
    remains in ``cumulative`` but cannot make the estimated steady-state rate
    look slower or faster.
    """

    completed_updates: int
    timing_sample_count: int
    validation_timing_sample_count: int
    support_satisfied: bool
    prior_attempt_cumulative_seconds: float
    screen_wall_seconds: float
    cumulative_validation_seconds: float
    cumulative_training_seconds: float
    envelope_limit_seconds: float
    mutation_limit_seconds: float
    remaining_mutation_seconds: float
    conservative_update_seconds: float | None
    conservative_validation_seconds: float | None
    projected_remaining_validation_seconds: float | None
    projected_minimum_training_seconds: float | None
    projected_minimum_envelope_seconds: float | None
    minimum_update_feasible: bool | None
    limit_hit_reason: LimitHitReason

    @property
    def limit_hit(self) -> bool:
        return self.limit_hit_reason != "none"


@dataclass(frozen=True, slots=True)
class SecondAttemptAdmissionEvidence:
    """Authenticated proof that a provisional second campaign can finish.

    This is deliberately not constructible from a caller's remaining-time
    estimate.  ``from_sealed_execution_timing`` consumes the exact v3 executor
    timing payload and binds it to both screen records, the first architecture
    attempt, the active progress record, and every validation boundary.  The
    caller must also supply this record's independently persisted digest to
    :func:`choose_second_attempt`.

    The current qualification coordinator has no provisional second-attempt
    execution phase, so it cannot produce this evidence yet and must pass
    ``None``.  A future coordinator may admit the full second campaign only
    after it has charged the second screen and collected timing support through
    a complete validation boundary.
    """

    schema: str
    protocol_sha256: str
    source_sha256: str
    first_attempt_sha256: str
    first_screen_result_sha256: str
    second_screen_result_sha256: str
    execution_progress_record_sha256: str
    execution_timing_evidence_sha256: str
    validation_timing_sha256: str
    architecture_attempt_sha256: str
    architecture_choice: SecondAttempt
    base_config_sha256: str
    resolved_config_sha256: str
    campaign_config_sha256: str
    first_screen_wall_seconds: float
    prior_attempt_cumulative_seconds: float
    second_screen_wall_seconds: float
    discarded_attempt_seconds: float
    completed_update_seconds: tuple[float, ...]
    completed_validation_seconds: tuple[float, ...]
    validation_boundary_updates: tuple[int, ...]
    minimum_updates: int
    validation_interval_updates: int
    required_validation_count: int
    reserved_audit_seconds: float
    projection: DynamicSetTimeProjection
    evidence_sha256: str

    @classmethod
    def from_sealed_execution_timing(
        cls,
        *,
        execution_timing: Mapping[str, object],
        validation_boundary_updates: Sequence[int],
        protocol_sha256: str,
        source_sha256: str,
        first_attempt_sha256: str,
        first_screen_result_sha256: str,
        first_screen_wall_seconds: float,
        second_screen_result_sha256: str,
        execution_progress_record_sha256: str,
        config: DynamicSetCampaignConfig = DEFAULT_CAMPAIGN,
    ) -> SecondAttemptAdmissionEvidence:
        """Reconstruct an admission record from one sealed executor payload."""

        config.validate()
        expected_timing_keys = {
            "schema",
            "architecture_attempt_index",
            "architecture_choice",
            "base_config_sha256",
            "resolved_config_sha256",
            "prior_attempt_cumulative_seconds",
            "architecture_attempt_sha256",
            "completed_update_seconds",
            "discarded_attempt_seconds",
            "screen_wall_seconds",
            "completed_validation_seconds",
            "validation_timing_sha256",
            "cumulative_training_seconds",
            "evidence_sha256",
        }
        if type(execution_timing) is not dict or set(execution_timing) != expected_timing_keys:
            raise ValueError("second-attempt execution timing schema differs")
        timing_body = {
            name: value for name, value in execution_timing.items() if name != "evidence_sha256"
        }
        if (
            execution_timing["schema"] != _EXECUTION_TIMING_EVIDENCE_SCHEMA
            or canonical_sha256(timing_body) != execution_timing["evidence_sha256"]
        ):
            raise ValueError("second-attempt execution timing digest differs")
        if type(execution_timing["completed_update_seconds"]) is not list:
            raise TypeError("second-attempt update timings must be an exact list")
        if type(execution_timing["completed_validation_seconds"]) is not list:
            raise TypeError("second-attempt validation timings must be an exact list")
        updates = tuple(execution_timing["completed_update_seconds"])
        validations = tuple(execution_timing["completed_validation_seconds"])
        boundaries = tuple(validation_boundary_updates)
        projection = project_minimum_update_feasibility(
            completed_update_seconds=updates,
            discarded_attempt_seconds=execution_timing["discarded_attempt_seconds"],
            prior_attempt_cumulative_seconds=execution_timing["prior_attempt_cumulative_seconds"],
            screen_wall_seconds=execution_timing["screen_wall_seconds"],
            completed_validation_seconds=validations,
            config=config,
        )
        unsigned: dict[str, object] = {
            "schema": _SECOND_ATTEMPT_ADMISSION_SCHEMA,
            "protocol_sha256": protocol_sha256,
            "source_sha256": source_sha256,
            "first_attempt_sha256": first_attempt_sha256,
            "first_screen_result_sha256": first_screen_result_sha256,
            "second_screen_result_sha256": second_screen_result_sha256,
            "execution_progress_record_sha256": execution_progress_record_sha256,
            "execution_timing_evidence_sha256": execution_timing["evidence_sha256"],
            "validation_timing_sha256": execution_timing["validation_timing_sha256"],
            "architecture_attempt_sha256": execution_timing["architecture_attempt_sha256"],
            "architecture_choice": execution_timing["architecture_choice"],
            "base_config_sha256": execution_timing["base_config_sha256"],
            "resolved_config_sha256": execution_timing["resolved_config_sha256"],
            "campaign_config_sha256": canonical_sha256(asdict(config)),
            "first_screen_wall_seconds": first_screen_wall_seconds,
            "prior_attempt_cumulative_seconds": execution_timing[
                "prior_attempt_cumulative_seconds"
            ],
            "second_screen_wall_seconds": execution_timing["screen_wall_seconds"],
            "discarded_attempt_seconds": execution_timing["discarded_attempt_seconds"],
            "completed_update_seconds": updates,
            "completed_validation_seconds": validations,
            "validation_boundary_updates": boundaries,
            "minimum_updates": config.minimum_updates,
            "validation_interval_updates": config.validation_interval_updates,
            "required_validation_count": (
                config.minimum_updates // config.validation_interval_updates
            ),
            "reserved_audit_seconds": config.reserved_audit_seconds,
            "projection": asdict(projection),
        }
        evidence = cls(
            **{name: value for name, value in unsigned.items() if name != "projection"},
            projection=projection,
            evidence_sha256=canonical_sha256(unsigned),
        )
        return evidence.validate(
            expected_evidence_sha256=evidence.evidence_sha256,
            config=config,
        )

    def _unsigned(self) -> dict[str, object]:
        return {name: value for name, value in asdict(self).items() if name != "evidence_sha256"}

    def _execution_timing_body(self) -> dict[str, object]:
        return {
            "schema": _EXECUTION_TIMING_EVIDENCE_SCHEMA,
            "architecture_attempt_index": 2,
            "architecture_choice": self.architecture_choice,
            "base_config_sha256": self.base_config_sha256,
            "resolved_config_sha256": self.resolved_config_sha256,
            "prior_attempt_cumulative_seconds": self.prior_attempt_cumulative_seconds,
            "architecture_attempt_sha256": self.architecture_attempt_sha256,
            "completed_update_seconds": list(self.completed_update_seconds),
            "discarded_attempt_seconds": self.discarded_attempt_seconds,
            "screen_wall_seconds": self.second_screen_wall_seconds,
            "completed_validation_seconds": list(self.completed_validation_seconds),
            "validation_timing_sha256": self.validation_timing_sha256,
            "cumulative_training_seconds": self.projection.cumulative_training_seconds,
        }

    def validate(
        self,
        *,
        expected_evidence_sha256: str,
        config: DynamicSetCampaignConfig = DEFAULT_CAMPAIGN,
    ) -> SecondAttemptAdmissionEvidence:
        """Replay the sealed timing/accounting record under an external digest."""

        if type(self) is not SecondAttemptAdmissionEvidence:
            raise TypeError("second-attempt admission requires its exact evidence type")
        config.validate()
        validated_sha256(expected_evidence_sha256, label="expected second-attempt admission")
        if expected_evidence_sha256 == _ZERO_SHA256:
            raise ValueError("second-attempt admission cannot use the zero digest")
        if self.schema != _SECOND_ATTEMPT_ADMISSION_SCHEMA:
            raise ValueError("second-attempt admission evidence schema differs")
        for name in (
            "protocol_sha256",
            "source_sha256",
            "first_attempt_sha256",
            "first_screen_result_sha256",
            "second_screen_result_sha256",
            "execution_progress_record_sha256",
            "execution_timing_evidence_sha256",
            "architecture_attempt_sha256",
            "base_config_sha256",
            "resolved_config_sha256",
            "campaign_config_sha256",
            "evidence_sha256",
        ):
            value = validated_sha256(getattr(self, name), label=name)
            if value == _ZERO_SHA256:
                raise ValueError(f"second-attempt {name} cannot use the zero digest")
        validated_sha256(
            self.validation_timing_sha256,
            label="validation_timing_sha256",
        )
        if bool(self.completed_validation_seconds) is (
            self.validation_timing_sha256 == _ZERO_SHA256
        ):
            raise ValueError("second-attempt validation timing digest presence differs")
        if self.evidence_sha256 != expected_evidence_sha256:
            raise ValueError("second-attempt admission differs from its campaign binding")
        if (
            self.architecture_choice not in {"widen_perception", "widen_relation"}
            or self.base_config_sha256 == self.resolved_config_sha256
            or self.first_attempt_sha256 == self.architecture_attempt_sha256
            or self.first_screen_result_sha256 == self.second_screen_result_sha256
        ):
            raise ValueError("second-attempt architecture binding differs")
        if self.campaign_config_sha256 != canonical_sha256(asdict(config)):
            raise ValueError("second-attempt evidence differs from the campaign config")
        for name in (
            "first_screen_wall_seconds",
            "prior_attempt_cumulative_seconds",
            "second_screen_wall_seconds",
            "discarded_attempt_seconds",
            "reserved_audit_seconds",
        ):
            value = getattr(self, name)
            if type(value) is not float:
                raise TypeError(f"second-attempt {name} must be an exact float")
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"second-attempt {name} must be finite and nonnegative")
        if self.first_screen_wall_seconds <= 0.0 or self.second_screen_wall_seconds <= 0.0:
            raise ValueError("both architecture screens require measured positive wall time")
        if self.prior_attempt_cumulative_seconds < self.first_screen_wall_seconds:
            raise ValueError("prior-attempt cumulative time omits its first screen")
        if type(self.completed_update_seconds) is not tuple or any(
            type(value) is not float for value in self.completed_update_seconds
        ):
            raise TypeError("second-attempt update timings must be an exact float tuple")
        if type(self.completed_validation_seconds) is not tuple or any(
            type(value) is not float for value in self.completed_validation_seconds
        ):
            raise TypeError("second-attempt validation timings must be an exact float tuple")
        if type(self.validation_boundary_updates) is not tuple or any(
            type(value) is not int or value <= 0 for value in self.validation_boundary_updates
        ):
            raise TypeError("second-attempt validation boundaries must be positive exact integers")
        expected_boundaries = tuple(
            config.validation_interval_updates * index
            for index in range(1, len(self.completed_validation_seconds) + 1)
        )
        if (
            len(self.completed_update_seconds) > config.maximum_updates
            or len(self.validation_boundary_updates) != len(self.completed_validation_seconds)
            or self.validation_boundary_updates != expected_boundaries
            or (
                self.validation_boundary_updates
                and self.validation_boundary_updates[-1] > len(self.completed_update_seconds)
            )
        ):
            raise ValueError("second-attempt validation timing lineage differs")
        expected_required_validations = config.minimum_updates // config.validation_interval_updates
        if (
            type(self.minimum_updates) is not int
            or self.minimum_updates != config.minimum_updates
            or type(self.validation_interval_updates) is not int
            or self.validation_interval_updates != config.validation_interval_updates
            or type(self.required_validation_count) is not int
            or self.required_validation_count != expected_required_validations
            or self.reserved_audit_seconds != config.reserved_audit_seconds
        ):
            raise ValueError("second-attempt minimum/validation/reserve accounting differs")
        reconstructed = project_minimum_update_feasibility(
            completed_update_seconds=self.completed_update_seconds,
            discarded_attempt_seconds=self.discarded_attempt_seconds,
            prior_attempt_cumulative_seconds=self.prior_attempt_cumulative_seconds,
            screen_wall_seconds=self.second_screen_wall_seconds,
            completed_validation_seconds=self.completed_validation_seconds,
            config=config,
        )
        if (
            type(self.projection) is not DynamicSetTimeProjection
            or self.projection != reconstructed
        ):
            raise ValueError("second-attempt projection differs from sealed raw timing evidence")
        if canonical_sha256(self._execution_timing_body()) != self.execution_timing_evidence_sha256:
            raise ValueError("second-attempt execution timing digest differs")
        if canonical_sha256(self._unsigned()) != self.evidence_sha256:
            raise ValueError("second-attempt admission evidence digest mismatch")
        return self

    def minimum_campaign_feasible(
        self,
        *,
        expected_evidence_sha256: str,
        config: DynamicSetCampaignConfig = DEFAULT_CAMPAIGN,
    ) -> bool:
        """Require supported, validation-complete timing below the 60h envelope."""

        self.validate(
            expected_evidence_sha256=expected_evidence_sha256,
            config=config,
        )
        completed = self.projection.completed_updates
        validation_count = len(self.completed_validation_seconds)
        return bool(
            completed >= config.minimum_timing_support_updates
            and completed % config.validation_interval_updates == 0
            and validation_count == completed // config.validation_interval_updates
            and self.projection.support_satisfied
            and self.projection.minimum_update_feasible is True
            and self.projection.limit_hit_reason == "none"
            and completed < config.minimum_updates
            and self.projection.projected_minimum_envelope_seconds is not None
            and self.projection.projected_minimum_envelope_seconds
            <= config.maximum_training_hours * 3600.0
        )


def _nearest_rank_percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


def _conservative_timing_rate(
    values: Sequence[float],
    *,
    config: DynamicSetCampaignConfig,
) -> float:
    mean = math.fsum(values) / len(values)
    if len(values) > 1:
        variance = math.fsum((sample - mean) ** 2 for sample in values) / (len(values) - 1)
        standard_error = math.sqrt(max(variance, 0.0) / len(values))
    else:
        standard_error = 0.0
    upper_mean = mean + config.timing_projection_confidence_z * standard_error
    p90 = _nearest_rank_percentile(values, 0.90)
    return max(upper_mean, p90) * config.timing_projection_safety_factor


def project_minimum_update_feasibility(
    *,
    completed_update_seconds: Sequence[float],
    discarded_attempt_seconds: float = 0.0,
    prior_attempt_cumulative_seconds: float = 0.0,
    screen_wall_seconds: float = 0.0,
    completed_validation_seconds: Sequence[float] = (),
    config: DynamicSetCampaignConfig = DEFAULT_CAMPAIGN,
) -> DynamicSetTimeProjection:
    """Project whether the supported minimum still fits before audit reserve.

    Only executor-measured evidence belongs here.  The function accepts no
    free-standing elapsed-time override: cumulative time is reconstructed from
    the immutable cumulative duration of any preceding architecture attempt,
    the active attempt's sealed screen duration, exact completed-update and
    validation samples, and measured discarded attempts.
    """

    config.validate()
    scalar_durations: dict[str, float] = {}
    for name, value in (
        ("discarded_attempt_seconds", discarded_attempt_seconds),
        ("prior_attempt_cumulative_seconds", prior_attempt_cumulative_seconds),
        ("screen_wall_seconds", screen_wall_seconds),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a finite scalar")
        duration = float(value)
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
        scalar_durations[name] = duration
    discarded = scalar_durations["discarded_attempt_seconds"]
    prior_attempt = scalar_durations["prior_attempt_cumulative_seconds"]
    screen = scalar_durations["screen_wall_seconds"]
    if len(completed_update_seconds) > config.maximum_updates:
        raise ValueError("timing evidence exceeds the campaign update bound")
    samples: list[float] = []
    for value in completed_update_seconds:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("completed-update timings must be finite scalars")
        sample = float(value)
        if not math.isfinite(sample) or sample < 0.0:
            raise ValueError("completed-update timings must be finite and nonnegative")
        samples.append(sample)

    validation_samples: list[float] = []
    for value in completed_validation_seconds:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("completed-validation timings must be finite scalars")
        sample = float(value)
        if not math.isfinite(sample) or sample < 0.0:
            raise ValueError("completed-validation timings must be finite and nonnegative")
        validation_samples.append(sample)

    completed = len(samples)
    maximum_completed_validations = completed // config.validation_interval_updates
    if len(validation_samples) > maximum_completed_validations:
        raise ValueError("validation timing evidence exceeds completed validation boundaries")
    if len(validation_samples) > config.maximum_updates // config.validation_interval_updates:
        raise ValueError("validation timing evidence exceeds the campaign bound")
    cumulative_validation = float(math.fsum(validation_samples))
    cumulative = float(math.fsum((prior_attempt, screen, *samples, discarded, *validation_samples)))
    mutation_limit = config.training_mutation_seconds
    # Authenticated timing can be repartitioned across hundreds of samples.
    # Treat a reconstruction within a few representable floats of the exact
    # policy boundary as exhausted so summation layout cannot reopen mutation.
    boundary_tolerance = 4.0 * math.ulp(mutation_limit)
    reserve_boundary_reached = cumulative >= mutation_limit - boundary_tolerance
    remaining = 0.0 if reserve_boundary_reached else mutation_limit - cumulative
    required_minimum_validations = config.minimum_updates // config.validation_interval_updates
    remaining_validations = max(0, required_minimum_validations - len(validation_samples))
    validation_support = bool(validation_samples) or remaining_validations == 0
    support = completed >= config.minimum_timing_support_updates and validation_support
    conservative_rate: float | None = None
    conservative_validation_rate: float | None = None
    projected_validation: float | None = None
    projected: float | None = None
    projected_envelope: float | None = None
    feasible: bool | None = None
    if support:
        window = samples[-config.timing_projection_window_updates :]
        conservative_rate = _conservative_timing_rate(window, config=config)
        conservative_validation_rate = (
            0.0
            if not validation_samples
            else _conservative_timing_rate(validation_samples, config=config)
        )
        projected_validation = float(conservative_validation_rate * remaining_validations)
        remaining_updates = max(0, config.minimum_updates - completed)
        projected = float(cumulative + conservative_rate * remaining_updates + projected_validation)
        projected_envelope = float(projected + config.reserved_audit_seconds)
        feasible = projected_envelope <= config.maximum_training_hours * 3600.0

    if reserve_boundary_reached:
        reason: LimitHitReason = "training_reserve_boundary"
    elif (
        support
        and feasible is False
        and (completed < config.minimum_updates or remaining_validations > 0)
    ):
        reason = "minimum_update_projection_infeasible"
    else:
        reason = "none"
    return DynamicSetTimeProjection(
        completed_updates=completed,
        timing_sample_count=completed,
        validation_timing_sample_count=len(validation_samples),
        support_satisfied=support,
        prior_attempt_cumulative_seconds=prior_attempt,
        screen_wall_seconds=screen,
        cumulative_validation_seconds=cumulative_validation,
        cumulative_training_seconds=cumulative,
        envelope_limit_seconds=config.maximum_training_hours * 3600.0,
        mutation_limit_seconds=mutation_limit,
        remaining_mutation_seconds=remaining,
        conservative_update_seconds=conservative_rate,
        conservative_validation_seconds=conservative_validation_rate,
        projected_remaining_validation_seconds=projected_validation,
        projected_minimum_training_seconds=projected,
        projected_minimum_envelope_seconds=projected_envelope,
        minimum_update_feasible=feasible,
        limit_hit_reason=reason,
    )


def learning_rate_multiplier(
    completed_update: int,
    config: DynamicSetCampaignConfig = DEFAULT_CAMPAIGN,
) -> float:
    config.validate()
    if isinstance(completed_update, bool) or not isinstance(completed_update, int):
        raise TypeError("completed_update must be an integer")
    if completed_update < 0 or completed_update > config.maximum_updates:
        raise ValueError("completed_update lies outside the campaign")
    if completed_update == 0:
        return 0.0
    if completed_update <= config.warmup_updates:
        return completed_update / config.warmup_updates
    decay_updates = config.maximum_updates - config.warmup_updates
    progress = (completed_update - config.warmup_updates) / decay_updates
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (
        config.final_learning_rate_fraction + (1.0 - config.final_learning_rate_fraction) * cosine
    )


def disposable_screen_failures(metrics: DisposableScreenMetrics) -> tuple[str, ...]:
    failures: list[str] = []
    if (
        isinstance(metrics.example_count, bool)
        or not isinstance(metrics.example_count, int)
        or metrics.example_count != 64
    ):
        failures.append("example_count:!=64")
    for name in (
        "initial_optimization_objective",
        "final_optimization_objective",
        "initial_objective_regret",
        "final_objective_regret",
        "proposal_f1",
        "collision_f1",
    ):
        if not math.isfinite(getattr(metrics, name)):
            failures.append(f"{name}:nonfinite")
    try:
        expected_initial_regret = dynamic_set_objective_regret(
            metrics.initial_optimization_objective
        )
        expected_final_regret = dynamic_set_objective_regret(metrics.final_optimization_objective)
    except ValueError:
        failures.append("optimization_objective:below_lower_bound")
    else:
        if not math.isclose(
            metrics.initial_objective_regret,
            expected_initial_regret,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ) or not math.isclose(
            metrics.final_objective_regret,
            expected_final_regret,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            failures.append("objective_regret:binding_mismatch")
    if metrics.initial_objective_regret <= 0.0:
        failures.append("initial_objective_regret:nonpositive")
    elif metrics.final_objective_regret / metrics.initial_objective_regret > 0.20:
        failures.append("objective_reduction:<80%")
    if metrics.proposal_f1 < 0.95:
        failures.append("proposal_f1:<0.95")
    if metrics.collision_f1 < 0.95:
        failures.append("collision_f1:<0.95")
    if not metrics.finite_owner_gradients:
        failures.append("finite_owner_gradients:false")
    if metrics.rejected_update_count != 0:
        failures.append("rejected_update_count:nonzero")
    if not 0 < metrics.completed_updates <= DEFAULT_CAMPAIGN.screen_maximum_updates:
        failures.append("completed_updates:outside_screen_budget")
    return tuple(failures)


def choose_second_attempt(
    *,
    first_attempt_failed_early: bool,
    oracle_state_dynamics_passed: bool,
    perception_gates_passed: bool,
    truth_state_contact_owns_error: bool,
    admission_evidence: SecondAttemptAdmissionEvidence | None = None,
    admission_evidence_sha256: str | None = None,
    config: DynamicSetCampaignConfig = DEFAULT_CAMPAIGN,
) -> SecondAttempt:
    """Choose a widening only from independently bound provisional evidence.

    Until the formal coordinator can run and seal the provisional timing phase,
    both evidence arguments are absent and this intentionally returns ``none``.
    """

    config.validate()
    if (admission_evidence is None) is not (admission_evidence_sha256 is None):
        return "none"
    if admission_evidence is not None and admission_evidence_sha256 is not None:
        if type(admission_evidence) is not SecondAttemptAdmissionEvidence:
            raise TypeError("second-attempt admission requires exact measured evidence")
        admission_evidence.validate(
            expected_evidence_sha256=admission_evidence_sha256,
            config=config,
        )
    if (
        not first_attempt_failed_early
        or admission_evidence is None
        or admission_evidence_sha256 is None
        or not admission_evidence.minimum_campaign_feasible(
            expected_evidence_sha256=admission_evidence_sha256,
            config=config,
        )
    ):
        return "none"
    if not perception_gates_passed and oracle_state_dynamics_passed:
        choice: SecondAttempt = "widen_perception"
    elif perception_gates_passed and truth_state_contact_owns_error:
        choice = "widen_relation"
    else:
        choice = "none"
    if admission_evidence.architecture_choice != choice:
        return "none"
    return choice


def configure_second_attempt(
    base_config: object,
    *,
    first_attempt_failed_early: bool,
    oracle_state_dynamics_passed: bool,
    perception_gates_passed: bool,
    truth_state_contact_owns_error: bool,
    admission_evidence: SecondAttemptAdmissionEvidence | None = None,
    admission_evidence_sha256: str | None = None,
    campaign_config: DynamicSetCampaignConfig = DEFAULT_CAMPAIGN,
) -> tuple[SecondAttempt, object | None]:
    """Return the sole permitted width change, or no second-attempt config.

    This combines evidence routing with construction so a caller cannot ask
    for both wider modules or widen a module without the declared diagnosis.
    The resulting model is still checked against all parameter/byte limits by
    the optimizer constructor.
    """

    from world_model.training.dynamic_set_config import OrpheusConfig

    if not isinstance(base_config, OrpheusConfig):
        raise TypeError("base_config must be an OrpheusConfig")
    base_config.validate()
    if (
        base_config.model.rgbd.observation_mode != "set"
        or base_config.model.rgbd.set_feature_dim != 32
        or base_config.model.dynamics.hidden_dim != 16
        or base_config.model.dynamics.relation_hidden_dim is not None
    ):
        raise ValueError("second-attempt construction requires the exact attempt-one widths")
    choice = choose_second_attempt(
        first_attempt_failed_early=first_attempt_failed_early,
        oracle_state_dynamics_passed=oracle_state_dynamics_passed,
        perception_gates_passed=perception_gates_passed,
        truth_state_contact_owns_error=truth_state_contact_owns_error,
        admission_evidence=admission_evidence,
        admission_evidence_sha256=admission_evidence_sha256,
        config=campaign_config,
    )
    if choice == "none":
        return choice, None
    if choice == "widen_perception":
        model = replace(
            base_config.model,
            rgbd=replace(base_config.model.rgbd, set_feature_dim=64),
        )
    else:
        model = replace(
            base_config.model,
            dynamics=replace(base_config.model.dynamics, relation_hidden_dim=64),
        )
    configured = replace(base_config, model=model)
    configured.validate()
    if admission_evidence is None:
        raise AssertionError("admitted second attempt lacks its evidence")
    if canonical_sha256(base_config.to_dict()) != admission_evidence.base_config_sha256:
        raise ValueError("second-attempt base configuration differs from its evidence")
    if canonical_sha256(configured.to_dict()) != admission_evidence.resolved_config_sha256:
        raise ValueError("second-attempt resolved configuration differs from its evidence")
    return choice, configured


def decide_dynamic_set_campaign(
    inspection: CampaignInspection,
    *,
    absolute_and_promotion_gates_passed: bool,
    failed_to_improve: bool = False,
    elapsed_training_hours: float = 0.0,
    execution_limit_hit_reason: LimitHitReason = "none",
    config: DynamicSetCampaignConfig = DEFAULT_CAMPAIGN,
) -> DynamicSetCampaignDecision:
    config.validate()
    if (
        isinstance(elapsed_training_hours, bool)
        or not isinstance(elapsed_training_hours, (int, float))
        or not math.isfinite(float(elapsed_training_hours))
        or elapsed_training_hours < 0.0
    ):
        raise ValueError("elapsed_training_hours must be finite and nonnegative")
    if type(execution_limit_hit_reason) is not str or execution_limit_hit_reason not in {
        "none",
        "training_reserve_boundary",
        "minimum_update_projection_infeasible",
    }:
        raise ValueError("execution_limit_hit_reason differs from the frozen schema")
    if (
        execution_limit_hit_reason == "training_reserve_boundary"
        and elapsed_training_hours < config.training_mutation_hours
    ):
        raise ValueError("reserve-boundary status precedes the training mutation limit")
    if (
        execution_limit_hit_reason != "none"
        or elapsed_training_hours >= config.training_mutation_hours
    ):
        if execution_limit_hit_reason == "minimum_update_projection_infeasible":
            reason = (
                "measured update timing cannot reach the supported minimum while "
                "preserving the frozen audit reserve"
            )
        else:
            reason = (
                f"the training mutation budget ended at the "
                f"{config.training_mutation_hours:g}-hour audit-reserve boundary "
                "without qualified convergence"
            )
        return DynamicSetCampaignDecision(
            status="limit_hit",
            reason=reason,
            next_total_updates=None,
        )
    if inspection.completed_steps > config.maximum_updates:
        raise ValueError("completed updates exceed the frozen campaign maximum")
    if inspection.completed_steps >= config.minimum_updates:
        completed_extension_updates = inspection.completed_steps - config.minimum_updates
        if completed_extension_updates % config.extension_updates:
            next_extension_boundary = (
                config.minimum_updates
                + (completed_extension_updates // config.extension_updates + 1)
                * config.extension_updates
            )
            return DynamicSetCampaignDecision(
                status="continue",
                reason=(
                    f"the current {config.validation_interval_updates:,}-update validation "
                    f"is inside an incomplete {config.extension_updates:,}-update extension"
                ),
                next_total_updates=next_extension_boundary,
            )
    # The campaign must complete the declared minimum, while checkpoint
    # selection may legitimately preserve an earlier, safer incumbent.
    if inspection.completed_steps >= config.minimum_updates and absolute_and_promotion_gates_passed:
        return DynamicSetCampaignDecision(
            status="qualified_convergence",
            reason=(
                "the campaign reached its minimum supported updates and the selected "
                "incumbent passed every promotion gate"
            ),
            next_total_updates=None,
        )
    decision = decide_continuation(
        inspection,
        minimum_total_steps=config.minimum_updates,
        extension_steps=config.extension_updates,
        tail_steps=config.plateau_validation_count * config.validation_interval_updates,
        minimum_relative_gain=config.minimum_relative_gain,
        maximum_total_steps=config.maximum_updates,
        plateau_validation_count=config.plateau_validation_count,
        validation_interval_steps=config.validation_interval_updates,
    )
    if inspection.completed_steps >= config.maximum_updates:
        return DynamicSetCampaignDecision(
            status="limit_hit",
            reason=(
                "the hard update cap was reached without qualified convergence; "
                "this is a budget stop"
            ),
            next_total_updates=None,
        )
    if decision.status == "plateau" and failed_to_improve:
        return DynamicSetCampaignDecision(
            status="failed_to_improve",
            reason=(
                "the objective reached its frozen plateau while the paired "
                "baseline-improvement requirement still failed"
            ),
            next_total_updates=None,
        )
    if decision.status == "plateau":
        return DynamicSetCampaignDecision(
            status="objective_plateau",
            reason=decision.reason,
            next_total_updates=None,
        )
    if decision.status == "limit_hit":
        return DynamicSetCampaignDecision(
            status="limit_hit",
            reason=decision.reason,
            next_total_updates=None,
        )
    if decision.status == "incomplete":
        return DynamicSetCampaignDecision(
            status="continue",
            reason=decision.reason,
            next_total_updates=config.minimum_updates,
        )
    return DynamicSetCampaignDecision(
        status="continue",
        reason=decision.reason,
        next_total_updates=decision.next_total_steps,
    )


__all__ = [
    "DEFAULT_CAMPAIGN",
    "DisposableScreenMetrics",
    "DynamicSetCampaignConfig",
    "DynamicSetCampaignDecision",
    "DynamicSetTimeProjection",
    "LimitHitReason",
    "SecondAttemptAdmissionEvidence",
    "TRAINING_CACHE_FULL_COVERAGE_UPDATES",
    "TRAINING_CACHE_TIMING_SUPPORT_UPDATES",
    "TRAINING_CACHE_WARM_WINDOW_START_UPDATES",
    "choose_second_attempt",
    "configure_second_attempt",
    "decide_dynamic_set_campaign",
    "disposable_screen_failures",
    "learning_rate_multiplier",
    "project_minimum_update_feasibility",
]
