"""Reversible tokenization of explicit predictive abstractions."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor

from world_model.abstractions.contracts import (
    AbstractionAssignment,
    BeliefTokenSchema,
    PredictiveTokenBatch,
    PredictiveTokenType,
)
from world_model.abstractions.router import PredictiveAbstractionRouter
from world_model.belief import (
    WorldBelief,
    pack_fast_state,
    pack_slow_state,
    unpack_fast_state,
    unpack_slow_state,
)


def _schema(belief: WorldBelief) -> BeliefTokenSchema:
    objects = belief.objects
    modal_parameter_dim = objects.modal_count * objects.modal_dim
    kinematic_width = 2 * objects.fast_state_dim
    program_width = (
        2 * objects.slow_state_dim + 2 * modal_parameter_dim + objects.parameter_memory.shape[-1]
    )
    lifecycle_width = 4 + objects.motion_mode_logits.shape[-1]
    scene_width = 3 + belief.global_code.shape[-1] + belief.global_log_variance.shape[-1]
    camera_width = 31 + belief.camera.log_variance.shape[-1]
    return BeliefTokenSchema(
        max_objects=objects.max_objects,
        fast_state_dim=objects.fast_state_dim,
        slow_state_dim=objects.slow_state_dim,
        modal_parameter_dim=modal_parameter_dim,
        parameter_memory_dim=objects.parameter_memory.shape[-1],
        motion_mode_dim=objects.motion_mode_logits.shape[-1],
        global_code_dim=belief.global_code.shape[-1],
        global_variance_dim=belief.global_log_variance.shape[-1],
        camera_variance_dim=belief.camera.log_variance.shape[-1],
        token_width=max(
            kinematic_width,
            program_width,
            lifecycle_width,
            scene_width,
            camera_width,
        ),
    )


def _write_payload(destination: Tensor, payload: Tensor) -> None:
    if payload.shape[:-1] != destination.shape[:-1]:
        raise ValueError("token payload prefix does not match destination")
    if payload.shape[-1] > destination.shape[-1]:
        raise ValueError("token payload exceeds configured token width")
    destination[..., : payload.shape[-1]] = payload


class WorldBeliefTokenizer:
    """Create transformer-ready typed tokens without changing belief truth.

    This adapter is parameter-free and reversible.  A learned input projection
    or transformer can consume the returned token sequence later, but its
    outputs must still be decoded and assimilated through the belief contracts.
    """

    def __init__(self, router: PredictiveAbstractionRouter | None = None) -> None:
        self.router = router or PredictiveAbstractionRouter()

    def encode(
        self,
        belief: WorldBelief,
        assignment: AbstractionAssignment | None = None,
    ) -> PredictiveTokenBatch:
        belief.validate()
        selected = assignment or self.router.route(belief)
        selected.validate()
        if selected.kind.shape != belief.objects.object_id.shape:
            raise ValueError("abstraction assignment does not match belief slots")
        schema = _schema(belief)
        batch = belief.batch_size
        values = belief.timestamp.new_zeros(
            batch,
            schema.sequence_length,
            schema.token_width,
        )
        valid = torch.zeros(
            batch,
            schema.sequence_length,
            device=belief.device,
            dtype=torch.bool,
        )
        valid[:, :2] = True
        token_type = torch.empty(
            schema.sequence_length,
            device=belief.device,
            dtype=torch.int64,
        )
        object_slot = torch.full_like(token_type, -1)
        object_id = torch.full(
            (batch, schema.sequence_length),
            -1,
            device=belief.device,
            dtype=torch.int64,
        )
        abstraction_kind = torch.full_like(object_id, -1)

        token_type[0] = int(PredictiveTokenType.SCENE)
        token_type[1] = int(PredictiveTokenType.CAMERA)
        scene = torch.cat(
            (
                belief.gravity,
                belief.global_code,
                belief.global_log_variance,
            ),
            dim=-1,
        )
        _write_payload(values[:, 0], scene)
        camera = torch.cat(
            (
                belief.camera.world_from_camera.flatten(start_dim=1),
                belief.camera.linear_velocity,
                belief.camera.angular_velocity,
                belief.camera.intrinsics.flatten(start_dim=1),
                belief.camera.log_variance,
            ),
            dim=-1,
        )
        _write_payload(values[:, 1], camera)

        fast = torch.cat(
            (pack_fast_state(belief.objects), belief.objects.fast_log_variance),
            dim=-1,
        )
        slow = torch.cat(
            (
                pack_slow_state(belief.objects),
                belief.objects.slow_log_variance,
                belief.objects.modal_frequency.flatten(start_dim=2),
                belief.objects.modal_decay_raw.flatten(start_dim=2),
                belief.objects.parameter_memory,
            ),
            dim=-1,
        )
        lifecycle = torch.cat(
            (
                belief.objects.existence_logit.unsqueeze(-1),
                belief.objects.visibility_logit.unsqueeze(-1),
                belief.objects.age_steps.to(dtype=belief.dtype).unsqueeze(-1),
                belief.objects.missed_steps.to(dtype=belief.dtype).unsqueeze(-1),
                belief.objects.motion_mode_logits,
            ),
            dim=-1,
        )
        for slot in range(schema.max_objects):
            base = 2 + 3 * slot
            for offset, token_kind, payload in (
                (0, PredictiveTokenType.ENTITY_KINEMATIC, fast[:, slot]),
                (1, PredictiveTokenType.ENTITY_PROGRAM, slow[:, slot]),
                (2, PredictiveTokenType.ENTITY_LIFECYCLE, lifecycle[:, slot]),
            ):
                index = base + offset
                token_type[index] = int(token_kind)
                object_slot[index] = slot
                object_id[:, index] = belief.objects.object_id[:, slot]
                abstraction_kind[:, index] = selected.kind[:, slot]
                valid[:, index] = belief.objects.active[:, slot]
                _write_payload(values[:, index], payload)

        return PredictiveTokenBatch(
            values=values,
            valid_mask=valid,
            token_type=token_type,
            object_slot=object_slot,
            object_id=object_id,
            abstraction_kind=abstraction_kind,
            timestamp=belief.timestamp.clone(),
            next_object_id=belief.next_object_id.clone(),
            camera_calibrated=belief.camera.calibrated.clone(),
            schema=schema,
        ).validate()

    def decode(
        self,
        tokens: PredictiveTokenBatch,
        template: WorldBelief,
    ) -> WorldBelief:
        """Restore a belief from tokens using ``template`` for static metadata."""

        tokens.validate()
        template.validate()
        expected = _schema(template)
        if tokens.schema != expected:
            raise ValueError("token schema is incompatible with belief template")
        if tokens.values.shape[0] != template.batch_size:
            raise ValueError("token batch does not match belief template")

        values = tokens.values
        scene_width = 3 + expected.global_code_dim + expected.global_variance_dim
        scene = values[:, 0, :scene_width]
        gravity = scene[:, :3]
        global_start = 3
        global_stop = global_start + expected.global_code_dim
        global_code = scene[:, global_start:global_stop]
        global_log_variance = scene[:, global_stop:]

        camera_width = 31 + expected.camera_variance_dim
        camera_values = values[:, 1, :camera_width]
        camera_offset = 0
        world_from_camera = camera_values[:, camera_offset : camera_offset + 16].reshape(-1, 4, 4)
        camera_offset += 16
        linear_velocity = camera_values[:, camera_offset : camera_offset + 3]
        camera_offset += 3
        angular_velocity = camera_values[:, camera_offset : camera_offset + 3]
        camera_offset += 3
        intrinsics = camera_values[:, camera_offset : camera_offset + 9].reshape(-1, 3, 3)
        camera_offset += 9
        camera_log_variance = camera_values[
            :, camera_offset : camera_offset + expected.camera_variance_dim
        ]

        objects = template.objects.clone()
        fast = values.new_empty(
            template.batch_size,
            expected.max_objects,
            2 * expected.fast_state_dim,
        )
        program_width = (
            2 * expected.slow_state_dim
            + 2 * expected.modal_parameter_dim
            + expected.parameter_memory_dim
        )
        program = values.new_empty(
            template.batch_size,
            expected.max_objects,
            program_width,
        )
        lifecycle_width = 4 + expected.motion_mode_dim
        lifecycle = values.new_empty(
            template.batch_size,
            expected.max_objects,
            lifecycle_width,
        )
        active = torch.zeros_like(objects.active)
        object_id = torch.full_like(objects.object_id, -1)
        for slot in range(expected.max_objects):
            base = 2 + 3 * slot
            fast[:, slot] = values[:, base, : 2 * expected.fast_state_dim]
            program[:, slot] = values[:, base + 1, :program_width]
            lifecycle[:, slot] = values[:, base + 2, :lifecycle_width]
            active[:, slot] = tokens.valid_mask[:, base]
            object_id[:, slot] = torch.where(
                active[:, slot],
                tokens.object_id[:, base],
                torch.full_like(tokens.object_id[:, base], -1),
            )

        objects = unpack_fast_state(fast[..., : expected.fast_state_dim], objects)
        objects = objects.replace(
            fast_log_variance=fast[..., expected.fast_state_dim :],
        )
        slow_offset = 0
        objects = unpack_slow_state(
            program[..., slow_offset : slow_offset + expected.slow_state_dim],
            objects,
        )
        slow_offset += expected.slow_state_dim
        slow_log_variance = program[..., slow_offset : slow_offset + expected.slow_state_dim]
        slow_offset += expected.slow_state_dim
        modal_frequency = program[
            ..., slow_offset : slow_offset + expected.modal_parameter_dim
        ].reshape_as(objects.modal_frequency)
        slow_offset += expected.modal_parameter_dim
        modal_decay_raw = program[
            ..., slow_offset : slow_offset + expected.modal_parameter_dim
        ].reshape_as(objects.modal_decay_raw)
        slow_offset += expected.modal_parameter_dim
        parameter_memory = program[..., slow_offset : slow_offset + expected.parameter_memory_dim]
        objects = objects.replace(
            object_id=object_id,
            active=active,
            slow_log_variance=slow_log_variance,
            modal_frequency=modal_frequency,
            modal_decay_raw=modal_decay_raw,
            parameter_memory=parameter_memory,
            existence_logit=lifecycle[..., 0],
            visibility_logit=lifecycle[..., 1],
            age_steps=lifecycle[..., 2].round().to(dtype=torch.int64),
            missed_steps=lifecycle[..., 3].round().to(dtype=torch.int64),
            motion_mode_logits=lifecycle[..., 4:],
        )
        camera = replace(
            template.camera,
            world_from_camera=world_from_camera,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            intrinsics=intrinsics,
            log_variance=camera_log_variance,
            calibrated=tokens.camera_calibrated,
        )
        return template.replace(
            timestamp=tokens.timestamp,
            objects=objects,
            camera=camera,
            gravity=gravity,
            global_code=global_code,
            global_log_variance=global_log_variance,
            next_object_id=tokens.next_object_id,
        ).validate()
