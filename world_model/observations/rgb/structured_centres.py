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
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    label,
    maximum_filter,
    sobel,
)
from scipy.optimize import least_squares, linear_sum_assignment
from torch import Tensor

from world_model.observations.rgb.roi_updater import make_roi_grid, sample_rois


@dataclass(frozen=True)
class StructuredCentreOutput:
    """Foreground centroids aligned to proposal order."""

    centres: Tensor
    radius_pixels: Tensor
    valid_mask: Tensor
    ambiguous_mask: Tensor
    depth_valid_mask: Tensor
    component_count: Tensor


@dataclass(frozen=True)
class StructuredROICentreOutput:
    """Per-object foreground centroids and trustworthy scales inside RGB ROIs."""

    centres: Tensor
    radius_pixels: Tensor
    valid_mask: Tensor
    ambiguous_mask: Tensor
    ownership_margin: Tensor
    depth_valid_mask: Tensor
    component_pixel_count: Tensor


@dataclass(frozen=True)
class PhotometricDiscGeometryOutput:
    """Renderer-fitted apparent radii qualified from RGB residuals."""

    radius_pixels: Tensor
    valid_mask: Tensor
    fit_rms: Tensor


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
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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
    ambiguity: list[bool] = []
    depth_validity: list[bool] = []
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
        component_touches_boundary = bool(
            (pixel_x == 0).any()
            or (pixel_x == width - 1).any()
            or (pixel_y == 0).any()
            or (pixel_y == height - 1).any()
        )
        component_scale_ambiguous = len(peak_centres) > 1
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
            ambiguity.append(component_scale_ambiguous)
            depth_validity.append(not component_touches_boundary and not component_scale_ambiguous)
    if not centres:
        return (
            torch.empty((0, 2), dtype=torch.float32),
            torch.empty((0,), dtype=torch.float32),
            torch.empty((0,), dtype=torch.bool),
            torch.empty((0,), dtype=torch.bool),
        )
    return (
        torch.tensor(centres, dtype=torch.float32),
        torch.tensor(radii, dtype=torch.float32),
        torch.tensor(ambiguity, dtype=torch.bool),
        torch.tensor(depth_validity, dtype=torch.bool),
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
    ambiguous = torch.zeros_like(valid)
    depth_valid = torch.zeros_like(valid)
    counts = torch.zeros(
        proposal_centres.shape[0],
        device=proposal_centres.device,
        dtype=torch.int64,
    )
    for batch_index in range(image.shape[0]):
        (
            component_centres,
            component_radius_pixels,
            component_ambiguous,
            component_depth_valid,
        ) = _foreground_centres(
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
            ambiguous[batch_index, proposal_row] = component_ambiguous[component_column].to(
                device=ambiguous.device
            )
            depth_valid[batch_index, proposal_row] = component_depth_valid[component_column].to(
                device=depth_valid.device
            )
    return StructuredCentreOutput(
        centres=refined,
        radius_pixels=refined_radius_pixels,
        valid_mask=valid,
        ambiguous_mask=ambiguous,
        depth_valid_mask=depth_valid,
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
            ambiguous_mask=torch.zeros(
                (batch, 0),
                device=output_device,
                dtype=torch.bool,
            ),
            ownership_margin=proposal_centres.new_full(
                (batch, 0),
                torch.finfo(proposal_centres.dtype).max,
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
    # The source-conditioned fast path may reject a crop, but it must not let
    # a subpixel prior change switch ownership between two nearby objects.  A
    # second disconnected foreground component at the same nearest supported
    # distance is an unresolved local tie.  Leave that recovery to global
    # discovery instead of emitting a discontinuous, identity-bearing
    # residual measurement.
    selected_seed_distance_squared = flat_seed_scores.gather(
        dim=-1,
        index=seed_index.unsqueeze(-1),
    ).squeeze(-1)
    alternative_seed_scores = torch.where(
        seed_candidates & ~component,
        distance_squared,
        infinity,
    )
    alternative_seed_distance_squared = alternative_seed_scores.reshape(
        batch,
        objects,
        -1,
    ).amin(dim=-1)
    selected_seed_distance = selected_seed_distance_squared.clamp_min(0.0).sqrt()
    alternative_seed_distance = alternative_seed_distance_squared.clamp_min(0.0).sqrt()
    has_alternative = torch.isfinite(alternative_seed_distance_squared)
    ownership_margin = torch.where(
        has_seed & has_alternative,
        alternative_seed_distance - selected_seed_distance,
        torch.full_like(
            selected_seed_distance,
            torch.finfo(selected_seed_distance.dtype).max,
        ),
    )
    comparison_alternative_distance = torch.where(
        has_alternative,
        alternative_seed_distance,
        selected_seed_distance,
    )
    tie_scale = torch.maximum(
        torch.maximum(selected_seed_distance, comparison_alternative_distance),
        torch.ones_like(selected_seed_distance),
    )
    tie_tolerance = 32.0 * torch.finfo(distance_squared.dtype).eps * tie_scale
    ambiguous = has_seed & has_alternative & (ownership_margin <= tie_tolerance)
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
        & ~ambiguous
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
    ambiguous = ambiguous.to(device=output_device)
    ownership_margin = ownership_margin.to(device=output_device, dtype=output_dtype)
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
        ambiguous_mask=ambiguous,
        ownership_margin=ownership_margin,
        depth_valid_mask=depth_valid,
        component_pixel_count=component_pixel_count,
    )


def _photometric_disc_fit(
    image: np.ndarray,
    row_background: np.ndarray,
    foreground_strength: np.ndarray,
    proposal_xy: tuple[float, float],
    *,
    threshold: float,
    maximum_fit_rms: float,
) -> tuple[float, float] | None:
    """Fit the known sphere image profile without using simulator labels."""

    height, width = foreground_strength.shape
    foreground = foreground_strength > threshold
    components, component_count = label(
        foreground,
        structure=np.ones((3, 3), dtype=np.int8),
    )
    if component_count == 0:
        return None
    proposal_x, proposal_y = proposal_xy
    foreground_y, foreground_x = np.nonzero(foreground)
    nearest = int(
        np.argmin(np.square(foreground_x - proposal_x) + np.square(foreground_y - proposal_y))
    )
    component_index = int(components[foreground_y[nearest], foreground_x[nearest]])
    component = components == component_index
    if int(component.sum()) < 8:
        return None
    if component[0].any() or component[-1].any() or component[:, 0].any() or component[:, -1].any():
        return None

    boundary_band = binary_dilation(component, iterations=1) & ~binary_erosion(
        component,
        iterations=1,
    )
    gradient_x = sobel(foreground_strength, axis=1, mode="nearest") / 8.0
    gradient_y = sobel(foreground_strength, axis=0, mode="nearest") / 8.0
    gradient = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    band_gradient = gradient[boundary_band]
    if band_gradient.size < 12 or float(band_gradient.max()) <= 0.0:
        return None
    selected_edge = boundary_band & (gradient >= 0.35 * float(band_gradient.max()))
    edge_y, edge_x = np.nonzero(selected_edge)
    if edge_x.size < 12:
        return None
    points = np.column_stack((edge_x, edge_y)).astype(np.float64)
    weights = gradient[selected_edge].astype(np.float64)
    design = np.column_stack(
        (
            2.0 * points[:, 0],
            2.0 * points[:, 1],
            np.ones(points.shape[0]),
        )
    )
    target = np.square(points).sum(axis=1)
    root_weight = np.sqrt(weights / weights.max())
    solution, *_ = np.linalg.lstsq(
        design * root_weight[:, None],
        target * root_weight,
        rcond=None,
    )
    centre_x, centre_y, constant = (float(value) for value in solution)
    radius_squared = constant + centre_x * centre_x + centre_y * centre_y
    if not math.isfinite(radius_squared) or radius_squared <= 0.0:
        return None
    radius = math.sqrt(radius_squared)
    radial = np.sqrt(np.square(points[:, 0] - centre_x) + np.square(points[:, 1] - centre_y))
    residual_pixels = float(np.sqrt(np.average(np.square(radial - radius), weights=weights)))
    angles = np.mod(
        np.arctan2(points[:, 1] - centre_y, points[:, 0] - centre_x),
        2.0 * math.pi,
    )
    angular_bins = np.floor(angles * 24.0 / (2.0 * math.pi)).astype(np.int64).clip(0, 23)
    angular_coverage = float(np.unique(angular_bins).size / 24.0)
    centre_distance = math.hypot(centre_x - proposal_x, centre_y - proposal_y)
    if radius < 1.5 or residual_pixels > 0.6 or angular_coverage < 0.75 or centre_distance > radius:
        return None

    margin = 2.0
    left = max(0, int(math.floor(centre_x - radius - margin)))
    right = min(width, int(math.ceil(centre_x + radius + margin + 1.0)))
    top = max(0, int(math.floor(centre_y - radius - margin)))
    bottom = min(height, int(math.ceil(centre_y + radius + margin + 1.0)))
    if left == 0 or top == 0 or right == width or bottom == height:
        return None
    pixel_y, pixel_x = np.meshgrid(
        np.arange(top, bottom, dtype=np.float64),
        np.arange(left, right, dtype=np.float64),
        indexing="ij",
    )
    observed = image[:, top:bottom, left:right]
    local_background = np.broadcast_to(row_background[:, top:bottom], observed.shape)
    centre_pixel = image[
        :,
        int(round(centre_y)),
        int(round(centre_x)),
    ]
    initial_albedo = np.clip(centre_pixel, 0.05, 1.0)

    def _render(parameters: np.ndarray) -> np.ndarray:
        candidate_x, candidate_y, candidate_radius = parameters[:3]
        albedo = parameters[3:, None, None]
        radial_squared = (
            np.square(pixel_x - candidate_x) + np.square(pixel_y - candidate_y)
        ) / max(candidate_radius * candidate_radius, 1.0e-8)
        complete_disc = radial_squared <= 1.0
        radial_distance = np.sqrt(np.maximum(radial_squared, 0.0))
        soft_width = 1.0 / max(candidate_radius, 1.0e-4)
        soft_support = np.clip((1.0 - radial_distance) / soft_width + 0.5, 0.0, 1.0) * complete_disc
        shade = np.clip(
            0.48 + 0.52 * np.sqrt(np.maximum(1.0 - radial_squared, 0.0)),
            0.0,
            1.0,
        )
        alpha = soft_support[None]
        return local_background * (1.0 - alpha) + albedo * shade[None] * alpha

    def _residual(parameters: np.ndarray) -> np.ndarray:
        return (_render(parameters) - observed).reshape(-1)

    lower = np.asarray(
        (
            centre_x - 0.75,
            centre_y - 0.75,
            max(1.25, radius - 0.75),
            0.0,
            0.0,
            0.0,
        ),
        dtype=np.float64,
    )
    upper = np.asarray(
        (
            centre_x + 0.75,
            centre_y + 0.75,
            radius + 0.75,
            1.0,
            1.0,
            1.0,
        ),
        dtype=np.float64,
    )
    fit = least_squares(
        _residual,
        np.concatenate(((centre_x, centre_y, radius), initial_albedo)),
        bounds=(lower, upper),
        method="trf",
        max_nfev=80,
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    fit_radius = float(fit.x[2])
    fit_centre_distance = math.hypot(
        float(fit.x[0]) - proposal_x,
        float(fit.x[1]) - proposal_y,
    )
    fit_rms = float(np.sqrt(np.mean(np.square(fit.fun))))
    if (
        not fit.success
        or not math.isfinite(fit_radius)
        or not math.isfinite(fit_rms)
        or fit_radius < 1.25
        or fit_centre_distance > fit_radius
        or fit_rms > maximum_fit_rms
    ):
        return None
    return fit_radius, fit_rms


def photometric_disc_geometry(
    image: Tensor,
    proposal_centres: Tensor,
    *,
    valid_mask: Tensor,
    threshold: float = 0.04,
    maximum_fit_rms: float = 0.035,
) -> PhotometricDiscGeometryOutput:
    """Recover complete apparent sphere scale from the known RGB image model.

    The fit consumes only RGB pixels and a source-conditioned image-space
    prior. It runs detached on CPU and rejects partial/merged silhouettes by
    the residual of the renderer's exact support and radial shading profile.
    Callers can retain learned gradients with a straight-through residual.
    """

    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("photometric disc input must have shape [B,3,H,W]")
    if proposal_centres.ndim != 3 or proposal_centres.shape[-1] != 2:
        raise ValueError("proposal_centres must have shape [B,N,2]")
    if image.shape[0] != proposal_centres.shape[0]:
        raise ValueError("image and proposal batch dimensions must match")
    if valid_mask.shape != proposal_centres.shape[:2] or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean [B,N]")
    if not 0.0 < threshold < 2.0:
        raise ValueError("photometric disc threshold must lie in (0,2)")
    if not math.isfinite(maximum_fit_rms) or maximum_fit_rms <= 0.0:
        raise ValueError("maximum_fit_rms must be finite and positive")

    output_radius = proposal_centres.new_zeros(proposal_centres.shape[:2])
    output_rms = proposal_centres.new_full(
        proposal_centres.shape[:2],
        torch.finfo(proposal_centres.dtype).max,
    )
    output_valid = torch.zeros_like(valid_mask)
    height, width = image.shape[-2:]
    cpu_image = image.detach().to(device="cpu", dtype=torch.float64)
    cpu_proposals = proposal_centres.detach().to(device="cpu", dtype=torch.float64)
    cpu_valid = valid_mask.detach().to(device="cpu")
    for batch_index in range(image.shape[0]):
        image_array = cpu_image[batch_index].numpy()
        row_background = np.median(image_array, axis=-1, keepdims=True)
        foreground_strength = np.linalg.norm(image_array - row_background, axis=0)
        for slot in torch.nonzero(cpu_valid[batch_index], as_tuple=False).flatten().tolist():
            proposal = cpu_proposals[batch_index, slot]
            if not bool(torch.isfinite(proposal).all()):
                continue
            proposal_pixels = (
                0.5 * (float(proposal[0]) + 1.0) * max(width - 1, 1),
                0.5 * (float(proposal[1]) + 1.0) * max(height - 1, 1),
            )
            fit = _photometric_disc_fit(
                image_array,
                row_background,
                foreground_strength,
                proposal_pixels,
                threshold=threshold,
                maximum_fit_rms=maximum_fit_rms,
            )
            if fit is None:
                continue
            radius, fit_rms = fit
            output_radius[batch_index, slot] = radius
            output_rms[batch_index, slot] = fit_rms
            output_valid[batch_index, slot] = True
    return PhotometricDiscGeometryOutput(
        radius_pixels=output_radius,
        valid_mask=output_valid,
        fit_rms=output_rms,
    )


__all__ = [
    "PhotometricDiscGeometryOutput",
    "StructuredCentreOutput",
    "StructuredROICentreOutput",
    "photometric_disc_geometry",
    "structured_disc_centres",
    "structured_disc_centres_in_rois",
]
