from __future__ import annotations

import torch

from world_model.observations.rgb.soft_geometry import soft_disc_geometry_from_rgb


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

