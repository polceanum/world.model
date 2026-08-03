"""Explicit gated Hungarian data association."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from world_model.fusion.costs import build_cost_matrix
from world_model.observations.measurements import MeasurementSet, PredictedMeasurements

if TYPE_CHECKING:
    from world_model.belief.world_belief import WorldBelief

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:  # pragma: no cover - dependency validation catches this
    raise ImportError("Project Orpheus association requires scipy") from exc


@dataclass
class AssociationResult:
    belief_indices: Tensor
    measurement_indices: Tensor
    pair_mask: Tensor
    pair_cost: Tensor
    unmatched_beliefs: Tensor
    unmatched_measurements: Tensor
    ambiguous: Tensor

    def validate(self) -> None:
        if self.belief_indices.shape != self.measurement_indices.shape:
            raise ValueError("association pair index shapes must match")
        if self.pair_mask.shape != self.belief_indices.shape:
            raise ValueError("association pair mask shape must match indices")
        if self.pair_cost.shape != self.belief_indices.shape:
            raise ValueError("association pair cost shape must match indices")
        if self.ambiguous.shape != self.belief_indices.shape:
            raise ValueError("association ambiguity shape must match indices")
        if self.pair_mask.dtype != torch.bool:
            raise TypeError("association pair_mask must be torch.bool")
        batch = self.pair_mask.shape[0]
        for batch_index in range(batch):
            mask = self.pair_mask[batch_index]
            beliefs = self.belief_indices[batch_index, mask]
            measurements = self.measurement_indices[batch_index, mask]
            if beliefs.unique().numel() != beliefs.numel():
                raise ValueError("a belief was assigned more than once")
            if measurements.unique().numel() != measurements.numel():
                raise ValueError("a measurement was assigned more than once")


class Associator:
    """Small-object-count CPU Hungarian matcher with uncertainty gating."""

    def __init__(
        self,
        *,
        geometry_weight: float = 1.0,
        appearance_weight: float = 0.25,
        existence_weight: float = 0.05,
        geometry_dimensions: int = 4,
        mahalanobis_gate: float = 25.0,
        maximum_cost: float = 30.0,
        ambiguity_margin: float = 0.5,
        minimum_measurement_confidence: float = 0.5,
    ) -> None:
        if not 0.0 <= minimum_measurement_confidence <= 1.0:
            raise ValueError("minimum_measurement_confidence must lie in [0,1]")
        if not math.isfinite(maximum_cost) or maximum_cost <= 0.0:
            raise ValueError("maximum_cost must be finite and positive")
        self.geometry_weight = geometry_weight
        self.appearance_weight = appearance_weight
        self.existence_weight = existence_weight
        self.geometry_dimensions = geometry_dimensions
        self.mahalanobis_gate = mahalanobis_gate
        self.maximum_cost = maximum_cost
        self.ambiguity_margin = ambiguity_margin
        self.minimum_measurement_confidence = minimum_measurement_confidence

    def cost_matrix(
        self,
        measured: MeasurementSet,
        predicted: PredictedMeasurements,
    ) -> Tensor:
        cost, _ = build_cost_matrix(
            measured,
            predicted,
            geometry_weight=self.geometry_weight,
            appearance_weight=self.appearance_weight,
            existence_weight=self.existence_weight,
            geometry_dimensions=self.geometry_dimensions,
            mahalanobis_gate=self.mahalanobis_gate,
            minimum_measurement_confidence=self.minimum_measurement_confidence,
        )
        source_belief_indices = measured.source_belief_indices
        source_object_ids = measured.source_object_ids
        if source_belief_indices is None and source_object_ids is None:
            return cost
        if not isinstance(source_belief_indices, Tensor) or not isinstance(
            source_object_ids,
            Tensor,
        ):
            raise ValueError(
                "source-conditioned measurements require belief indices and object IDs"
            )
        expected = measured.measurement_mask.shape
        if source_belief_indices.shape != expected or source_object_ids.shape != expected:
            raise ValueError("measurement source identity must have shape [B,M]")
        if (
            source_belief_indices.dtype is not torch.int64
            or source_object_ids.dtype is not torch.int64
        ):
            raise TypeError("measurement source identity must use int64")
        valid_sources = measured.measurement_mask
        if bool(torch.any(valid_sources & ((source_belief_indices < 0) | (source_object_ids < 0)))):
            raise ValueError("valid source-conditioned measurements require nonnegative identity")
        same_slot = predicted.belief_indices[:, :, None] == source_belief_indices[:, None, :]
        same_identity = predicted.object_ids[:, :, None] == source_object_ids[:, None, :]
        return cost.masked_fill(~(same_slot & same_identity), torch.inf)

    def match(
        self,
        belief: WorldBelief,
        measurements: MeasurementSet,
        predicted: PredictedMeasurements,
    ) -> AssociationResult:
        measurements.validate()
        predicted.validate()
        if belief.batch_size != predicted.values.shape[0]:
            raise ValueError("belief and predicted measurement batch sizes differ")
        cost = self.cost_matrix(measurements, predicted)
        batch, prediction_count, measurement_count = cost.shape
        belief_count = belief.objects.max_objects
        pair_capacity = min(prediction_count, measurement_count)
        device = cost.device
        belief_indices = torch.full((batch, pair_capacity), -1, dtype=torch.int64, device=device)
        measurement_indices = torch.full_like(belief_indices, -1)
        pair_mask = torch.zeros((batch, pair_capacity), dtype=torch.bool, device=device)
        pair_cost = torch.full((batch, pair_capacity), torch.inf, dtype=cost.dtype, device=device)
        ambiguous = torch.zeros_like(pair_mask)
        unmatched_beliefs = torch.zeros_like(belief.objects.active)
        unmatched_measurements = measurements.measurement_mask.clone()

        for batch_index in range(batch):
            valid_prediction_rows = torch.nonzero(
                predicted.valid_mask[batch_index],
                as_tuple=False,
            ).flatten()
            mapped_beliefs = predicted.belief_indices[
                batch_index,
                valid_prediction_rows,
            ]
            if bool(torch.any((mapped_beliefs < 0) | (mapped_beliefs >= belief_count))):
                raise ValueError("valid predicted rows must map to an in-range belief slot")
            if mapped_beliefs.unique().numel() != mapped_beliefs.numel():
                raise ValueError("valid predicted rows must map to unique belief slots")
            if mapped_beliefs.numel():
                if not bool(belief.objects.active[batch_index, mapped_beliefs].all()):
                    raise ValueError("valid predicted rows must map to active belief slots")
                mapped_ids = belief.objects.object_id[batch_index, mapped_beliefs]
                predicted_ids = predicted.object_ids[batch_index, valid_prediction_rows]
                if not torch.equal(mapped_ids, predicted_ids):
                    raise ValueError("predicted object IDs do not match their mapped belief slots")
                unmatched_beliefs[batch_index, mapped_beliefs] = True

        # Transfer before SciPy matching.  MPS does not implement float64, so
        # asking `.to(cpu, float64)` in one operation can attempt the cast on
        # the source device.  SciPy's matcher accepts CPU float32 directly.
        detached = cost.detach().to(device="cpu", dtype=torch.float32).numpy()
        for batch_index in range(batch):
            valid_prediction_rows = torch.nonzero(
                predicted.valid_mask[batch_index], as_tuple=False
            ).flatten()
            valid_measurements = torch.nonzero(
                measurements.measurement_mask[batch_index], as_tuple=False
            ).flatten()
            if valid_prediction_rows.numel() == 0 or valid_measurements.numel() == 0:
                continue
            prediction_np = valid_prediction_rows.cpu().numpy()
            measurement_np = valid_measurements.cpu().numpy()
            subcost = detached[batch_index][np.ix_(prediction_np, measurement_np)]
            admissible = np.isfinite(subcost) & (subcost <= self.maximum_cost)
            maximum_assignment_count = min(subcost.shape)
            # Gate impossible pairs before Hungarian assignment. SciPy rejects
            # infeasible all-inf rows/columns, so normalize valid costs to
            # [0,1] and assign every invalid edge a finite cost greater than
            # the sum of all possible valid edges. This is lexicographic:
            # maximize valid cardinality first, then minimize original cost.
            finite_subcost = np.full(
                subcost.shape,
                float(maximum_assignment_count + 1),
                dtype=np.float64,
            )
            if bool(admissible.any()):
                admissible_costs = subcost[admissible].astype(np.float64)
                minimum_admissible = float(admissible_costs.min())
                admissible_span = float(admissible_costs.max()) - minimum_admissible
                if admissible_span > 0.0:
                    finite_subcost[admissible] = (
                        admissible_costs - minimum_admissible
                    ) / admissible_span
                else:
                    finite_subcost[admissible] = 0.0
            rows, columns = linear_sum_assignment(finite_subcost)
            output_index = 0
            for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
                prediction_index = int(prediction_np[row])
                belief_index = int(predicted.belief_indices[batch_index, prediction_index].item())
                measurement_index = int(measurement_np[column])
                selected_cost = float(subcost[row, column])
                if not admissible[row, column] or output_index >= pair_capacity:
                    continue
                belief_indices[batch_index, output_index] = belief_index
                measurement_indices[batch_index, output_index] = measurement_index
                pair_mask[batch_index, output_index] = True
                pair_cost[batch_index, output_index] = cost[
                    batch_index, prediction_index, measurement_index
                ]
                competing = np.concatenate(
                    (
                        np.delete(np.where(admissible[row, :], subcost[row, :], np.inf), column),
                        np.delete(
                            np.where(admissible[:, column], subcost[:, column], np.inf),
                            row,
                        ),
                    )
                )
                finite_competing = competing[np.isfinite(competing)]
                second = float(finite_competing.min()) if finite_competing.size else float("inf")
                ambiguous[batch_index, output_index] = (
                    second - selected_cost < self.ambiguity_margin
                )
                unmatched_beliefs[batch_index, belief_index] = False
                unmatched_measurements[batch_index, measurement_index] = False
                output_index += 1

        result = AssociationResult(
            belief_indices=belief_indices,
            measurement_indices=measurement_indices,
            pair_mask=pair_mask,
            pair_cost=pair_cost,
            unmatched_beliefs=unmatched_beliefs,
            unmatched_measurements=unmatched_measurements,
            ambiguous=ambiguous,
        )
        result.validate()
        return result
