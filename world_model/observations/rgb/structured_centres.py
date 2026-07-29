"""Structured RGB centre proposals for the synthetic disc world.

The extractor uses only RGB pixels.  It estimates the static row-wise
background robustly, finds connected foreground components, and assigns their
centroids to learned proposal slots on the global path.  The fast path samples
only the projected object ROIs and refines each prior locally.  Both are
optional synthetic-world measurement priors behind the normal RGB contract,
not simulator-state input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, label, maximum_filter
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from world_model.observations.rgb.roi_updater import make_roi_grid, sample_rois


@dataclass(frozen=True)
class StructuredCentreOutput:
    """Foreground centroids aligned to proposal order."""

    centres: Tensor
    radius_pixels: Tensor
    valid_mask: Tensor
    component_count: Tensor


@dataclass(frozen=True)
class StructuredROICentreOutput:
    """Per-object foreground centroids and trustworthy scales inside RGB ROIs."""

    centres: Tensor
    radius_pixels: Tensor
    valid_mask: Tensor
    depth_valid_mask: Tensor
    component_pixel_count: Tensor


def _validate_controls(
    *,
    threshold: float,
    minimum_pixels: int,
    maximum_assignment_distance: float,
) -> None:
    if not 0 < threshold < 2:
        raise ValueError("structured disc threshold must lie in (0,2)")
    if minimum_pixels <= 0:
        raise ValueError("structured disc minimum_pixels must be positive")
    if maximum_assignment_distance <= 0:
        raise ValueError("structured disc maximum assignment distance must be positive")


def _foreground_centres(
    image: Tensor,
    *,
    threshold: float,
    minimum_pixels: int,
) -> tuple[Tensor, Tensor]:
    """Return normalized connected-component centroids for one ``[3,H,W]`` image."""

    _, height, width = image.shape
    cpu_image = image.detach().to(device="cpu", dtype=torch.float32)
    # Fewer than half of any row is occupied in the base toy distribution.
    # The per-row median therefore recovers the gradient/floor background
    # without importing renderer state into runtime.
    row_background = cpu_image.median(dim=-1).values.unsqueeze(-1)
    foreground_strength = torch.linalg.vector_norm(cpu_image - row_background, dim=0)
    foreground = np.asarray(foreground_strength > threshold)
    components, component_count = label(
        foreground,
        structure=np.ones((3, 3), dtype=np.int8),
    )
    centres: list[tuple[float, float]] = []
    radii: list[float] = []
    for component_index in range(1, component_count + 1):
        component_mask = components == component_index
        pixel_y, pixel_x = np.nonzero(component_mask)
        if pixel_x.size < minimum_pixels:
            continue
        # Two antialiased discs become one connected component as soon as
        # their silhouettes touch.  Distance-transform peaks retain the two
        # interiors, so split the component into nearest-peak basins before
        # taking photometric centroids.  This is still a direct RGB operation;
        # proposal state is used only by the later slot assignment.
        distance = distance_transform_edt(component_mask)
        peak_mask = (
            (distance == maximum_filter(distance, size=3, mode="constant"))
            & component_mask
            & (distance > 1.0)
        )
        peak_components, peak_count = label(
            peak_mask,
            structure=np.ones((3, 3), dtype=np.int8),
        )
        peak_centres: list[tuple[float, float]] = []
        for peak_index in range(1, peak_count + 1):
            peak_y, peak_x = np.nonzero(peak_components == peak_index)
            if peak_x.size:
                peak_centres.append((float(peak_y.mean()), float(peak_x.mean())))
        if not peak_centres:
            peak_centres.append((float(pixel_y.mean()), float(pixel_x.mean())))
        peaks = np.asarray(peak_centres, dtype=np.float32)
        pixels = np.stack((pixel_y, pixel_x), axis=-1).astype(np.float32)
        basin = np.square(pixels[:, None, :] - peaks[None, :, :]).sum(axis=-1).argmin(axis=1)
        for basin_index in range(peaks.shape[0]):
            selected = basin == basin_index
            basin_y = pixel_y[selected]
            basin_x = pixel_x[selected]
            if basin_x.size < minimum_pixels:
                continue
            y_index = torch.from_numpy(basin_y)
            x_index = torch.from_numpy(basin_x)
            weights = (foreground_strength[y_index, x_index] - float(threshold)).clamp_min(1.0e-6)
            centre_x = (x_index.to(weights.dtype) * weights).sum() / weights.sum()
            centre_y = (y_index.to(weights.dtype) * weights).sum() / weights.sum()
            normalized_x = 2.0 * centre_x / max(width - 1, 1) - 1.0
            normalized_y = 2.0 * centre_y / max(height - 1, 1) - 1.0
            centres.append((float(normalized_x), float(normalized_y)))
            radii.append(math.sqrt(float(basin_x.size) / math.pi))
    if not centres:
        return (
            torch.empty((0, 2), dtype=torch.float32),
            torch.empty((0,), dtype=torch.float32),
        )
    return (
        torch.tensor(centres, dtype=torch.float32),
        torch.tensor(radii, dtype=torch.float32),
    )


def structured_disc_centres(
    image: Tensor,
    proposal_centres: Tensor,
    *,
    threshold: float = 0.04,
    minimum_pixels: int = 4,
    maximum_assignment_distance: float = 0.75,
) -> StructuredCentreOutput:
    """Align RGB foreground centroids with learned proposal slots.

    The connected-component operation is intentionally a detached CPU global
    discovery primitive.  The returned tensors are restored to the input
    device/dtype; callers may use a straight-through residual so learned
    proposal heads continue receiving gradients.
    """

    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("structured disc input must have shape [B,3,H,W]")
    if proposal_centres.ndim != 3 or proposal_centres.shape[-1] != 2:
        raise ValueError("proposal_centres must have shape [B,Q,2]")
    if image.shape[0] != proposal_centres.shape[0]:
        raise ValueError("image and proposal batch dimensions must match")
    _validate_controls(
        threshold=threshold,
        minimum_pixels=minimum_pixels,
        maximum_assignment_distance=maximum_assignment_distance,
    )

    refined = proposal_centres.detach().clone()
    refined_radius_pixels = proposal_centres.new_zeros(proposal_centres.shape[:2])
    valid = torch.zeros(
        proposal_centres.shape[:2],
        device=proposal_centres.device,
        dtype=torch.bool,
    )
    counts = torch.zeros(
        proposal_centres.shape[0],
        device=proposal_centres.device,
        dtype=torch.int64,
    )
    for batch_index in range(image.shape[0]):
        component_centres, component_radius_pixels = _foreground_centres(
            image[batch_index],
            threshold=threshold,
            minimum_pixels=minimum_pixels,
        )
        counts[batch_index] = component_centres.shape[0]
        if component_centres.numel() == 0 or proposal_centres.shape[1] == 0:
            continue
        proposal_cpu = (
            proposal_centres[batch_index]
            .detach()
            .to(
                device="cpu",
                dtype=torch.float32,
            )
        )
        finite_rows = torch.isfinite(proposal_cpu).all(dim=-1)
        finite_row_indices = torch.nonzero(finite_rows, as_tuple=False).flatten()
        if finite_row_indices.numel() == 0:
            continue
        cost = torch.cdist(proposal_cpu[finite_row_indices], component_centres)
        proposal_rows, component_columns = linear_sum_assignment(np.asarray(cost))
        for finite_proposal_row, component_column in zip(
            proposal_rows,
            component_columns,
            strict=True,
        ):
            if float(cost[finite_proposal_row, component_column]) > maximum_assignment_distance:
                continue
            proposal_row = int(finite_row_indices[finite_proposal_row])
            refined[batch_index, proposal_row] = component_centres[component_column].to(
                device=refined.device,
                dtype=refined.dtype,
            )
            refined_radius_pixels[batch_index, proposal_row] = component_radius_pixels[
                component_column
            ].to(
                device=refined.device,
                dtype=refined.dtype,
            )
            valid[batch_index, proposal_row] = True
    return StructuredCentreOutput(
        centres=refined,
        radius_pixels=refined_radius_pixels,
        valid_mask=valid,
        component_count=counts,
    )


def structured_disc_centres_in_rois(
    image: Tensor,
    proposal_centres: Tensor,
    rois: Tensor,
    *,
    valid_mask: Tensor | None = None,
    output_size: int = 24,
    threshold: float = 0.04,
    minimum_pixels: int = 4,
    maximum_assignment_distance: float = 0.75,
) -> StructuredROICentreOutput:
    """Refine one centre per projected ROI using only the enclosed RGB pixels.

    Inputs use normalized image coordinates.  RGB crops and their coordinate
    grids are sampled in one batched ``grid_sample`` operation.  A component
    seeded at the nearest locally-supported foreground pixel is then grown
    with small tensor max-pools; this rejects isolated noise and prevents a
    fast update from rediscovering or globally assigning objects elsewhere in
    the frame.

    The operation is intentionally detached from autograd.  The caller can
    apply the result as a straight-through residual while supervising the raw
    learned centre separately.
    """

    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("structured ROI image must have shape [B,3,H,W]")
    if image.shape[-2] <= 0 or image.shape[-1] <= 0:
        raise ValueError("structured ROI image dimensions must be positive")
    if proposal_centres.ndim != 3 or proposal_centres.shape[-1] != 2:
        raise ValueError("proposal_centres must have shape [B,N,2]")
    if rois.ndim != 3 or rois.shape[-1] != 4:
        raise ValueError("rois must have shape [B,N,4]")
    if image.shape[0] != proposal_centres.shape[0] or rois.shape[:2] != (
        image.shape[0],
        proposal_centres.shape[1],
    ):
        raise ValueError("image, proposal, and ROI batch/object dimensions must match")
    if valid_mask is not None and valid_mask.shape != proposal_centres.shape[:2]:
        raise ValueError("valid_mask must have shape [B,N]")
    if output_size < 2:
        raise ValueError("structured ROI output_size must be at least two")
    _validate_controls(
        threshold=threshold,
        minimum_pixels=minimum_pixels,
        maximum_assignment_distance=maximum_assignment_distance,
    )

    batch, objects, _ = proposal_centres.shape
    output_device = proposal_centres.device
    output_dtype = proposal_centres.dtype
    if objects == 0:
        return StructuredROICentreOutput(
            centres=proposal_centres.detach().clone(),
            radius_pixels=proposal_centres.new_zeros((batch, 0)),
            valid_mask=torch.zeros(
                (batch, 0),
                device=output_device,
                dtype=torch.bool,
            ),
            depth_valid_mask=torch.zeros(
                (batch, 0),
                device=output_device,
                dtype=torch.bool,
            ),
            component_pixel_count=torch.zeros(
                (batch, 0),
                device=output_device,
                dtype=torch.int64,
            ),
        )

    sampling_rois = rois.detach().to(device=image.device, dtype=image.dtype)
    sampling_proposals = proposal_centres.detach().to(
        device=image.device,
        dtype=image.dtype,
    )
    roi_finite = torch.isfinite(sampling_rois).all(dim=-1)
    proposal_finite = torch.isfinite(sampling_proposals).all(dim=-1)
    safe_rois = torch.where(
        roi_finite.unsqueeze(-1),
        sampling_rois,
        torch.zeros_like(sampling_rois),
    )
    safe_proposals = torch.where(
        proposal_finite.unsqueeze(-1),
        sampling_proposals,
        torch.zeros_like(sampling_proposals),
    )
    roi_extent = safe_rois[..., 2:] - safe_rois[..., :2]
    usable_roi = roi_finite & proposal_finite & (roi_extent > 0).all(dim=-1)
    if valid_mask is not None:
        usable_roi = usable_roi & valid_mask.detach().to(
            device=image.device,
            dtype=torch.bool,
        )

    crops = sample_rois(
        image.detach(),
        safe_rois,
        output_size,
    )
    coordinate_grid = make_roi_grid(safe_rois, output_size)
    in_image = (coordinate_grid.abs() <= 1.0).all(dim=-1)

    # The projected sphere support normally touches its bounding ROI only at
    # four tangent points, so a median over the complete crop perimeter is a
    # robust local estimate of the row-gradient background.
    top = crops[..., 0, :]
    bottom = crops[..., -1, :]
    left = crops[..., 1:-1, 0]
    right = crops[..., 1:-1, -1]
    border = torch.cat((top, bottom, left, right), dim=-1)
    background = border.median(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
    foreground_strength = torch.linalg.vector_norm(crops - background, dim=2)
    foreground = (foreground_strength > threshold) & in_image

    flat_foreground = foreground.reshape(batch * objects, 1, output_size, output_size)
    neighbor_kernel = foreground_strength.new_ones((1, 1, 3, 3))
    neighbor_count = F.conv2d(
        flat_foreground.to(foreground_strength.dtype),
        neighbor_kernel,
        padding=1,
    ).reshape(batch, objects, output_size, output_size)
    # Four-neighbour local support is enough to reject isolated/small speckles;
    # the full grown component is still checked against ``minimum_pixels``.
    required_local_support = min(minimum_pixels, 4)
    seed_candidates = foreground & (neighbor_count >= required_local_support)
    distance_squared = (
        (coordinate_grid - safe_proposals.unsqueeze(-2).unsqueeze(-2)).square().sum(dim=-1)
    )
    infinity = torch.full_like(distance_squared, torch.inf)
    seed_scores = torch.where(seed_candidates, distance_squared, infinity)
    flat_seed_scores = seed_scores.reshape(batch, objects, -1)
    has_seed = torch.isfinite(flat_seed_scores).any(dim=-1)
    seed_index = flat_seed_scores.argmin(dim=-1)
    seed = F.one_hot(
        seed_index,
        num_classes=output_size * output_size,
    ).reshape(batch * objects, 1, output_size, output_size)
    reached = seed.to(torch.bool) & has_seed.reshape(batch * objects, 1, 1, 1)

    # A solid synthetic disc has a path diameter below 2*S.  Growing for this
    # fixed bound keeps the implementation vectorized over batch and objects
    # without invoking full-frame CPU connected-component analysis.
    for _ in range(2 * output_size):
        reached = (
            F.max_pool2d(reached.to(foreground_strength.dtype), 3, stride=1, padding=1) > 0
        ) & flat_foreground
    component = reached.reshape(batch, objects, output_size, output_size)
    component_pixel_count = component.sum(dim=(-1, -2), dtype=torch.int64)
    touches_crop_border = (
        component[..., 0, :].any(dim=-1)
        | component[..., -1, :].any(dim=-1)
        | component[..., :, 0].any(dim=-1)
        | component[..., :, -1].any(dim=-1)
    )

    weights = (foreground_strength - foreground_strength.new_tensor(threshold)).clamp_min(
        torch.finfo(foreground_strength.dtype).eps
    )
    weights = weights * component.to(weights.dtype)
    weight_sum = weights.sum(dim=(-1, -2))
    candidate_centres = (weights.unsqueeze(-1) * coordinate_grid).sum(
        dim=(-3, -2)
    ) / weight_sum.clamp_min(torch.finfo(weights.dtype).eps).unsqueeze(-1)
    assignment_distance = torch.linalg.vector_norm(
        candidate_centres - safe_proposals,
        dim=-1,
    )
    matched = (
        usable_roi
        & has_seed
        & (component_pixel_count >= minimum_pixels)
        & torch.isfinite(candidate_centres).all(dim=-1)
        & (assignment_distance <= maximum_assignment_distance)
    )
    # Convert sampled component area back to source-image pixels. A component
    # touching the crop edge has unknown missing area, so its centre may still
    # be useful but its scale must not drive monocular depth.
    source_height, source_width = image.shape[-2:]
    roi_width_pixels = roi_extent[..., 0] * (0.5 * max(source_width - 1, 1))
    roi_height_pixels = roi_extent[..., 1] * (0.5 * max(source_height - 1, 1))
    sampled_pixel_area = roi_width_pixels * roi_height_pixels / float(max(output_size - 1, 1) ** 2)
    radius_pixels = (
        (
            component_pixel_count.to(sampled_pixel_area.dtype)
            * sampled_pixel_area.clamp_min(0.0)
            / math.pi
        )
        .clamp_min(0.0)
        .sqrt()
    )
    depth_valid = matched & ~touches_crop_border & torch.isfinite(radius_pixels)

    candidate_centres = candidate_centres.to(
        device=output_device,
        dtype=output_dtype,
    )
    matched = matched.to(device=output_device)
    depth_valid = depth_valid.to(device=output_device)
    radius_pixels = radius_pixels.to(device=output_device, dtype=output_dtype)
    component_pixel_count = component_pixel_count.to(device=output_device)
    refined = torch.where(
        matched.unsqueeze(-1),
        candidate_centres,
        proposal_centres.detach(),
    )
    return StructuredROICentreOutput(
        centres=refined,
        radius_pixels=torch.where(
            depth_valid,
            radius_pixels,
            torch.zeros_like(radius_pixels),
        ),
        valid_mask=matched,
        depth_valid_mask=depth_valid,
        component_pixel_count=component_pixel_count,
    )


__all__ = [
    "StructuredCentreOutput",
    "StructuredROICentreOutput",
    "structured_disc_centres",
    "structured_disc_centres_in_rois",
]
