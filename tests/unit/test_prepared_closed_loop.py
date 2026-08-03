from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch
from torch import Tensor

from world_model.datasets import collate_episodes
from world_model.runtime import OnlineWorldModel
from world_model.simulator import generate_episode
from world_model.training.loop import TrainingBatchResult, run_closed_loop_batch
from world_model.utils.config import load_config


def _closed_loop_config() -> Any:
    config = load_config("configs/tiny_overfit.yaml")
    config = replace(
        config,
        simulator=replace(
            config.simulator,
            image_size=(24, 24),
            sequence_frames=4,
            min_objects=1,
            max_objects=1,
            camera_motion="fixed",
            render_noise_std=0.0,
        ),
        model=replace(
            config.model,
            rgb=replace(
                config.model.rgb,
                backbone_channels=(8, 16, 24, 32),
                feature_dim=16,
                proposal_queries=3,
                roi_size=8,
                global_every_steps=100,
                global_uncertainty_threshold=1.0e6,
                surprise_threshold=1.0e6,
            ),
            dynamics=replace(config.model.dynamics, hidden_dim=24),
            filter=replace(config.model.filter, hidden_dim=32),
            lifecycle=replace(
                config.model.lifecycle,
                birth_confidence=0.0,
                birth_confirmations=1,
            ),
        ),
        training=replace(
            config.training,
            batch_size=1,
            tbptt_steps=4,
            horizon_weights=(1.0,),
        ),
        evaluation=replace(
            config.evaluation,
            horizons_seconds=(0.05,),
            episodes=1,
        ),
    )
    config.validate()
    return config


def _assert_result_close(
    actual: TrainingBatchResult,
    expected: TrainingBatchResult,
) -> None:
    assert actual.phase == expected.phase
    torch.testing.assert_close(
        actual.total_loss,
        expected.total_loss,
        rtol=1.0e-6,
        atol=1.0e-7,
        equal_nan=True,
    )
    assert actual.loss_terms.keys() == expected.loss_terms.keys()
    for name in actual.loss_terms:
        torch.testing.assert_close(
            actual.loss_terms[name],
            expected.loss_terms[name],
            rtol=1.0e-6,
            atol=1.0e-7,
            equal_nan=True,
        )
    assert actual.metrics.keys() == expected.metrics.keys()
    for name in actual.metrics:
        torch.testing.assert_close(
            torch.as_tensor(actual.metrics[name]),
            torch.as_tensor(expected.metrics[name]),
            rtol=1.0e-6,
            atol=1.0e-7,
            equal_nan=True,
        )


def test_closed_loop_uses_one_propagation_per_noninitial_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _closed_loop_config()
    batch = collate_episodes([generate_episode(config, seed=7)])
    model = OnlineWorldModel.from_config(config, device="cpu")
    original_predict_step = model.dynamics.predict_step
    propagation_deltas: list[Tensor] = []

    def counting_predict_step(belief: Any, delta: Any) -> Any:
        propagation_deltas.append(torch.as_tensor(delta).detach().clone())
        return original_predict_step(belief, delta)

    monkeypatch.setattr(model.dynamics, "predict_step", counting_predict_step)
    run_closed_loop_batch(
        model,
        batch,
        config,
        window_steps=4,
        apply_perturbations=False,
        include_measurement_supervision=False,
        rollout_anchors_per_window=1,
        compute_future_correction=False,
    )

    # Initialization retains its exact zero-duration dynamics semantics.
    # Every later observation is propagated exactly once and carries real dt.
    assert len(propagation_deltas) == 4
    torch.testing.assert_close(propagation_deltas[0], torch.zeros(1))
    for delta in propagation_deltas[1:]:
        assert bool((delta > 0).all())


def test_prepared_closed_loop_matches_duplicate_reference_forward_and_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _closed_loop_config()
    batch = collate_episodes([generate_episode(config, seed=9)])
    torch.manual_seed(23)
    prepared_model = OnlineWorldModel.from_config(config, device="cpu")
    duplicate_reference = OnlineWorldModel.from_config(config, device="cpu")
    duplicate_reference.load_state_dict(prepared_model.state_dict())

    # This test-only adapter reproduces the former behavior: the loop prepares
    # one prior for supervision, then ingest ignores it and propagates the same
    # source a second time. It is an exact semantic reference, not an oracle.
    original_reference_ingest = duplicate_reference.ingest

    def duplicate_ingest(
        packets: Any,
        *,
        prepared: Any = None,
    ) -> Any:
        del prepared
        return original_reference_ingest(packets)

    monkeypatch.setattr(duplicate_reference, "ingest", duplicate_ingest)
    prepared_result = run_closed_loop_batch(
        prepared_model,
        batch,
        config,
        window_steps=3,
        apply_perturbations=False,
        include_measurement_supervision=True,
        rollout_anchors_per_window=1,
        compute_future_correction=False,
    )
    reference_result = run_closed_loop_batch(
        duplicate_reference,
        batch,
        config,
        window_steps=3,
        apply_perturbations=False,
        include_measurement_supervision=True,
        rollout_anchors_per_window=1,
        compute_future_correction=False,
    )

    _assert_result_close(prepared_result, reference_result)
    prepared_model.zero_grad(set_to_none=True)
    duplicate_reference.zero_grad(set_to_none=True)
    prepared_result.total_loss.backward()
    reference_result.total_loss.backward()
    prepared_parameters = dict(prepared_model.named_parameters())
    reference_parameters = dict(duplicate_reference.named_parameters())
    assert prepared_parameters.keys() == reference_parameters.keys()
    compared_gradients = 0
    for name in prepared_parameters:
        prepared_gradient = prepared_parameters[name].grad
        reference_gradient = reference_parameters[name].grad
        if prepared_gradient is None or reference_gradient is None:
            assert prepared_gradient is None and reference_gradient is None
            continue
        torch.testing.assert_close(
            prepared_gradient,
            reference_gradient,
            rtol=2.0e-5,
            atol=2.0e-6,
            equal_nan=True,
            msg=lambda message, parameter_name=name: f"{parameter_name}: {message}",
        )
        compared_gradients += 1
    assert compared_gradients > 0
