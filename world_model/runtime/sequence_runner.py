"""Causal episode runner that reuses the exact online ingest path."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from torch import Tensor, nn

from world_model.belief import BeliefTrajectory, WorldBelief
from world_model.observations import ObservationPacket
from world_model.runtime.diagnostics import RuntimeStepDiagnostics
from world_model.runtime.online_world_model import OnlineWorldModel


@dataclass
class SequenceStepOutput:
    belief: WorldBelief
    trajectory: BeliefTrajectory | None
    diagnostics: tuple[RuntimeStepDiagnostics, ...]


@dataclass
class SequenceOutput:
    steps: list[SequenceStepOutput]
    final_belief: WorldBelief


class OnlineSequenceRunner(nn.Module):
    """Run timestamped packet groups with configurable truncated BPTT."""

    def __init__(self, model: OnlineWorldModel) -> None:
        super().__init__()
        self.model = model

    def run_episode(
        self,
        packet_groups: Sequence[ObservationPacket | Sequence[ObservationPacket]],
        *,
        rollout_queries: Sequence[float] | Tensor | None = None,
        tbptt_steps: int | None = None,
    ) -> SequenceOutput:
        if not packet_groups:
            raise ValueError("episode requires at least one packet group")
        self.model.reset()
        outputs: list[SequenceStepOutput] = []
        diagnostic_offset = 0
        for step_index, packet_group in enumerate(packet_groups):
            belief = self.model.ingest(packet_group)
            trajectory = None if rollout_queries is None else self.model.predict(rollout_queries)
            current_diagnostics = tuple(self.model.diagnostics.records[diagnostic_offset:])
            diagnostic_offset = len(self.model.diagnostics.records)
            outputs.append(
                SequenceStepOutput(
                    belief=belief,
                    trajectory=trajectory,
                    diagnostics=current_diagnostics,
                )
            )
            if tbptt_steps is not None and tbptt_steps > 0 and (step_index + 1) % tbptt_steps == 0:
                self.model.detach_state()
        final = self.model.belief
        assert final is not None
        return SequenceOutput(steps=outputs, final_belief=final)

    forward = run_episode
