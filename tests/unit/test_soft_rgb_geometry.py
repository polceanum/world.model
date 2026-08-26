from __future__ import annotations

import torch

from world_model.observations.rgb.soft_geometry import (
    PHOTOMETRIC_CANDIDATES_PER_STAGE,
    PHOTOMETRIC_CENTRE_TRUST_STEPS_PIXELS,
    PHOTOMETRIC_DAMPING_FORMULA,
    PHOTOMETRIC_RADIUS_SCALE_BRACKET,
    PHOTOMETRIC_RESIDUAL_NORMALIZATION,
    PHOTOMETRIC_TRUST_TRANSFORM,
    soft_disc_geometry_from_rgb,
    soft_photometric_disc_radius,
)


def _renderer_like_rgb(
    *,
    size: int,
    discs: tuple[tuple[float, float, float, tuple[float, float, float]], ...],
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(size, dtype=dtype),
        torch.arange(size, dtype=dtype),
        indexing="ij",
    )
    vertical = pixel_y / (size - 1)
    image = torch.stack(
        (
            0.08 + 0.16 * vertical,
            0.11 + 0.18 * vertical,
            0.16 + 0.18 * vertical,
        )
    )
    for centre_x, centre_y, radius, colour in discs:
        distance = ((pixel_x - centre_x).square() + (pixel_y - centre_y).square()).sqrt()
        # Match the toy renderer's one-pixel antialiasing while retaining its
        # hard physical silhouette support.
        alpha = (radius - distance + 0.5).clamp(0.0, 1.0) * (distance <= radius)
        sphere = torch.tensor(colour, dtype=dtype).view(3, 1, 1)
        image = image * (1.0 - alpha) + sphere * alpha
    return image


def _shaded_renderer_profile(
    *,
    size: int,
    centre_x: float,
    centre_y: float,
    radius: float,
    colour: tuple[float, float, float],
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Render the public toy profile without simulator state or episode seeds."""

    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(size, dtype=dtype),
        torch.arange(size, dtype=dtype),
        indexing="ij",
    )
    vertical = pixel_y / (size - 1)
    image = torch.stack(
        (
            0.08 + 0.16 * vertical,
            0.11 + 0.18 * vertical,
            0.16 + 0.18 * vertical,
        )
    )
    row = torch.arange(size)
    bands = ((row % max(6, size // 8)) == 0) & (row > size // 2)
    image[:, bands, :] *= 0.82
    radial_squared = (
        (pixel_x - centre_x).square() + (pixel_y - centre_y).square()
    ) / radius**2
    complete_disc = radial_squared <= 1.0
    radial_distance = radial_squared.clamp_min(0.0).sqrt()
    alpha = (radius * (1.0 - radial_distance) + 0.5).clamp(0.0, 1.0)
    alpha = alpha * complete_disc
    shade = (0.48 + 0.52 * (1.0 - radial_squared).clamp_min(0.0).sqrt()).clamp(0.0, 1.0)
    sphere = torch.tensor(colour, dtype=dtype).view(3, 1, 1) * shade
    return image * (1.0 - alpha) + sphere * alpha


def _pixel_centres(centres: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    return torch.stack(
        (
            0.5 * (centres[..., 0] + 1.0) * (width - 1),
            0.5 * (centres[..., 1] + 1.0) * (height - 1),
        ),
        dim=-1,
    )


def test_soft_rgb_moments_recover_an_isolated_renderer_like_disc() -> None:
    size = 40
    expected_centre = torch.tensor((13.2, 22.4), dtype=torch.float64)
    expected_radius = 6.7
    image = _renderer_like_rgb(
        size=size,
        discs=((13.2, 22.4, expected_radius, (0.9, 0.2, 0.1)),),
    ).unsqueeze(0)

    output = soft_disc_geometry_from_rgb(image)
    measured_centre = _pixel_centres(output.centres, height=size, width=size)[0, 0]

    torch.testing.assert_close(measured_centre, expected_centre, rtol=0.0, atol=0.06)
    torch.testing.assert_close(
        output.radius_pixels[0, 0],
        torch.tensor(expected_radius, dtype=image.dtype),
        rtol=0.0,
        atol=0.04,
    )
    assert output.valid_mask.tolist() == [[True]]
    assert output.confidence[0, 0] > 0.99
    assert output.foreground_probability.shape == (1, 1, size, size)
    assert output.effective_masks.shape == (1, 1, size, size)


def test_soft_rgb_geometry_gradient_matches_central_finite_difference() -> None:
    size = 40
    image = _renderer_like_rgb(
        size=size,
        discs=(
            (10.0, 20.0, 4.5, (0.9, 0.2, 0.1)),
            (29.0, 20.0, 4.5, (0.2, 0.8, 0.3)),
        ),
    ).unsqueeze(0)
    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(size, dtype=image.dtype),
        torch.arange(size, dtype=image.dtype),
        indexing="ij",
    )

    def objective(mask_centre_x: torch.Tensor) -> torch.Tensor:
        logits = 2.0 - (
            (pixel_x - mask_centre_x).square() + (pixel_y - 20.0).square()
        ) / (2.0 * 7.0**2)
        output = soft_disc_geometry_from_rgb(image, logits[None, None])
        return (
            output.centres[..., 0].sum()
            + 0.02 * output.radius_pixels.sum()
            + 0.1 * output.confidence.sum()
        )

    mask_centre_x = torch.tensor(12.0, dtype=image.dtype, requires_grad=True)
    value = objective(mask_centre_x)
    value.backward()
    assert mask_centre_x.grad is not None
    analytic = mask_centre_x.grad.detach()
    step = 1.0e-4
    finite_difference = (
        objective(mask_centre_x.detach() + step) - objective(mask_centre_x.detach() - step)
    ) / (2.0 * step)

    assert analytic.abs() > 1.0e-3
    torch.testing.assert_close(analytic, finite_difference, rtol=2.0e-4, atol=1.0e-7)


def test_one_parameter_soft_mask_converges_to_rgb_only_disc_geometry() -> None:
    size = 40
    left_disc = (10.0, 20.0, 4.5, (0.9, 0.2, 0.1))
    isolated_left = _renderer_like_rgb(size=size, discs=(left_disc,)).unsqueeze(0)
    composite = _renderer_like_rgb(
        size=size,
        discs=(left_disc, (29.0, 20.0, 4.5, (0.2, 0.8, 0.3))),
    ).unsqueeze(0)
    with torch.no_grad():
        target = soft_disc_geometry_from_rgb(isolated_left)
        target_centre = target.centres.detach()
        target_radius = target.radius_pixels.detach()

    _, pixel_x = torch.meshgrid(
        torch.arange(size, dtype=composite.dtype),
        torch.arange(size, dtype=composite.dtype),
        indexing="ij",
    )
    slope = torch.nn.Parameter(torch.zeros((), dtype=composite.dtype))
    optimizer = torch.optim.Adam((slope,), lr=0.5)
    initial_loss: float | None = None
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        # The only learned quantity is a smooth left/right ownership mask.
        # Geometry comes from the RGB moments, not from a label or detached
        # connected-component result.
        logits = slope * (0.5 * (size - 1) - pixel_x)
        output = soft_disc_geometry_from_rgb(composite, logits[None, None])
        loss = (
            100.0 * (output.centres - target_centre).square().sum()
            + 0.01 * (output.radius_pixels - target_radius).square().sum()
        )
        if initial_loss is None:
            initial_loss = float(loss.detach())
        loss.backward()
        optimizer.step()

    assert initial_loss is not None
    assert float(loss.detach()) < initial_loss * 1.0e-8
    torch.testing.assert_close(output.centres, target_centre, rtol=0.0, atol=1.0e-7)
    torch.testing.assert_close(output.radius_pixels, target_radius, rtol=0.0, atol=1.0e-6)
    assert float(slope.detach()) > 1.0


def test_gauss_newton_geometry_recovers_100_public_renderer_profiles() -> None:
    size = 48
    centre_pixels = (
        (18.13, 20.87),
        (20.37, 23.61),
        (24.63, 19.39),
        (27.89, 25.11),
    )
    radii = (1.60, 1.85, 2.10, 2.40, 2.85)
    colours = (
        (0.90, 0.20, 0.12),
        (0.20, 0.88, 0.25),
        (0.18, 0.25, 0.92),
        (0.72, 0.68, 0.24),
        (0.60, 0.22, 0.75),
    )
    profiles = tuple(
        (centre_index, radius_index, colour_index, centre, radius, colour)
        for centre_index, centre in enumerate(centre_pixels)
        for radius_index, radius in enumerate(radii)
        for colour_index, colour in enumerate(colours)
    )
    image = torch.stack(
        tuple(
            _shaded_renderer_profile(
                size=size,
                centre_x=centre[0],
                centre_y=centre[1],
                radius=radius,
                colour=colour,
                dtype=torch.float32,
            )
            for _, _, _, centre, radius, colour in profiles
        )
    )
    expected_centres = image.new_tensor(
        tuple(centre for _, _, _, centre, _, _ in profiles)
    )
    expected_radii = image.new_tensor(
        tuple(radius for _, _, _, _, radius, _ in profiles)
    )
    proposal_offsets = image.new_tensor(
        tuple(
            (
                0.281 if (centre_index + radius_index) % 2 else -0.281,
                0.281 if (radius_index + colour_index) % 2 else -0.281,
            )
            for centre_index, radius_index, colour_index, _, _, _ in profiles
        )
    )
    proposal_scales = image.new_tensor(
        tuple(
            1.09 if (centre_index + colour_index) % 2 else 0.91
            for centre_index, _, colour_index, _, _, _ in profiles
        )
    )
    proposal_centres_pixels = expected_centres + proposal_offsets
    centre = torch.stack(
        (
            2.0 * proposal_centres_pixels[:, 0] / (size - 1) - 1.0,
            2.0 * proposal_centres_pixels[:, 1] / (size - 1) - 1.0,
        ),
        dim=-1,
    ).unsqueeze(1)
    proposal = (proposal_scales * expected_radii).unsqueeze(1)

    with torch.no_grad():
        output = soft_photometric_disc_radius(image, centre, proposal)
    measured_centres = _pixel_centres(
        output.centres,
        height=size,
        width=size,
    )[:, 0]
    centre_error = torch.linalg.vector_norm(
        measured_centres - expected_centres,
        dim=-1,
    )
    relative_error = (output.radius_pixels[:, 0] / expected_radii - 1.0).abs()

    assert len(profiles) == 100
    assert centre_error.max() < 0.025
    assert relative_error.max() < 0.011
    assert output.fit_mse.max().sqrt() < 6.0e-4
    assert output.stage_fit_ratio.max() < 0.87
    assert output.stage_max_trust_fraction.max() < 0.80
    assert output.stage_monotonic_mask.all()
    assert output.support_valid_mask.all()
    assert output.all_stage_finite_mask.all()
    assert output.final_fit_valid_mask.all()
    torch.testing.assert_close(
        output.valid_mask,
        output.all_stage_finite_mask & output.final_fit_valid_mask,
    )
    assert output.valid_mask.all()
    assert PHOTOMETRIC_CANDIDATES_PER_STAGE == 7
    assert PHOTOMETRIC_CENTRE_TRUST_STEPS_PIXELS == (0.5, 0.25, 0.125, 0.0625)
    assert PHOTOMETRIC_RADIUS_SCALE_BRACKET == (0.8, 1.2)
    assert PHOTOMETRIC_DAMPING_FORMULA == (
        "sqrt(dtype_epsilon)*mean(diag(J^T_J))+dtype_epsilon"
    )
    assert PHOTOMETRIC_TRUST_TRANSFORM == "componentwise_tanh"
    assert PHOTOMETRIC_RESIDUAL_NORMALIZATION == "sqrt(3*height*width)"


def test_gauss_newton_geometry_gradient_matches_central_difference() -> None:
    size = 48
    image = _shaded_renderer_profile(
        size=size,
        centre_x=21.8,
        centre_y=24.2,
        radius=2.4,
        colour=(0.82, 0.27, 0.64),
    ).unsqueeze(0)

    def objective(centre_x_pixels: torch.Tensor, log_radius: torch.Tensor) -> torch.Tensor:
        centre = torch.stack(
            (
                2.0 * centre_x_pixels / (size - 1) - 1.0,
                centre_x_pixels.new_tensor(2.0 * 24.0 / (size - 1) - 1.0),
            )
        ).reshape(1, 1, 2)
        output = soft_photometric_disc_radius(
            image,
            centre,
            log_radius.exp().reshape(1, 1),
        )
        fitted_centre_pixels = _pixel_centres(
            output.centres,
            height=size,
            width=size,
        )
        return (
            fitted_centre_pixels[..., 0].sum()
            + 0.25 * fitted_centre_pixels[..., 1].sum()
            + 0.1 * output.radius_pixels.log().sum()
        )

    centre_x_pixels = torch.tensor(22.05, dtype=image.dtype, requires_grad=True)
    log_radius = torch.tensor(2.25, dtype=image.dtype).log().requires_grad_(True)
    value = objective(centre_x_pixels, log_radius)
    value.backward()
    assert centre_x_pixels.grad is not None
    assert log_radius.grad is not None
    assert torch.isfinite(centre_x_pixels.grad)
    assert torch.isfinite(log_radius.grad)
    assert centre_x_pixels.grad.abs() > 1.0e-8
    assert log_radius.grad.abs() > 1.0e-8

    step = 1.0e-5
    centre_finite_difference = (
        objective(centre_x_pixels.detach() + step, log_radius.detach())
        - objective(centre_x_pixels.detach() - step, log_radius.detach())
    ) / (2.0 * step)
    radius_finite_difference = (
        objective(centre_x_pixels.detach(), log_radius.detach() + step)
        - objective(centre_x_pixels.detach(), log_radius.detach() - step)
    ) / (2.0 * step)
    torch.testing.assert_close(
        centre_x_pixels.grad,
        centre_finite_difference,
        rtol=2.0e-3,
        atol=2.0e-5,
    )
    torch.testing.assert_close(
        log_radius.grad,
        radius_finite_difference,
        rtol=2.0e-3,
        atol=2.0e-5,
    )
