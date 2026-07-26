"""One-architecture training curriculum stage selection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumState:
    name: str
    rgb_only: bool
    perception_only: bool
    perturbation_enabled: bool


def curriculum_for_step(step: int, *, rgb_pretrain_steps: int) -> CurriculumState:
    if step < 0:
        raise ValueError("step must be nonnegative")
    if step < rgb_pretrain_steps:
        return CurriculumState(
            name="rgb_measurement_pretraining",
            rgb_only=True,
            perception_only=True,
            perturbation_enabled=False,
        )
    return CurriculumState(
        name="closed_loop_rgb",
        rgb_only=True,
        perception_only=False,
        perturbation_enabled=True,
    )
