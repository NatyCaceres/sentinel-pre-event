from pathlib import Path

import numpy as np
import pytest

from src.spectral.patch import SpectralPatch
from src.spectral.visualization import (
    SpectralVisualizationError,
    calculate_channel_statistics,
    percentile_stretch,
    prepare_false_color,
    prepare_index,
    prepare_index_display,
    prepare_rgb,
)


CHANNELS = (
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "B12",
    "NDMI",
    "NBR",
)


def build_patch() -> SpectralPatch:
    data = np.zeros(
        (
            8,
            64,
            64,
        ),
        dtype=np.float32,
    )

    x = np.linspace(
        0.0,
        1.0,
        64,
        dtype=np.float32,
    )

    horizontal = np.tile(
        x,
        (64, 1),
    )

    vertical = horizontal.T

    diagonal = (
        horizontal
        + vertical
    ) / 2.0

    # Bandas visibles con patrones distintos.
    data[0] = diagonal      # B2
    data[1] = vertical      # B3
    data[2] = horizontal    # B4

    # Otras bandas.
    data[3] = 0.5
    data[4] = 0.3
    data[5] = 0.2

    # Índices.
    data[6] = np.linspace(
        -1.2,
        1.2,
        64 * 64,
        dtype=np.float32,
    ).reshape(
        64,
        64,
    )

    data[7] = 0.25

    return SpectralPatch(
        data=data,
        channels=CHANNELS,
        path=Path("synthetic.tif"),
    )


def test_percentile_stretch_returns_unit_range() -> None:
    channel = np.linspace(
        0.0,
        10.0,
        64 * 64,
        dtype=np.float32,
    ).reshape(
        64,
        64,
    )

    result = percentile_stretch(
        channel
    )

    assert result.shape == (
        64,
        64,
    )

    assert result.dtype == np.float32

    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


def test_invalid_percentiles_are_rejected() -> None:
    channel = np.zeros(
        (64, 64),
        dtype=np.float32,
    )

    with pytest.raises(
        SpectralVisualizationError
    ):
        percentile_stretch(
            channel,
            lower_percentile=98,
            upper_percentile=2,
        )


def test_prepare_rgb_has_expected_shape() -> None:
    patch = build_patch()

    rgb = prepare_rgb(
        patch
    )

    assert rgb.shape == (
        64,
        64,
        3,
    )

    assert rgb.dtype == np.float32

    assert float(rgb.min()) >= 0.0
    assert float(rgb.max()) <= 1.0


def test_prepare_rgb_preserves_channel_patterns() -> None:
    patch = build_patch()

    rgb = prepare_rgb(
        patch
    )

    # B4 era horizontal, por lo que el rojo
    # debe aumentar de izquierda a derecha.
    assert (
        rgb[32, 60, 0]
        > rgb[32, 5, 0]
    )

    # B3 era vertical, por lo que el verde
    # debe aumentar desde arriba hacia abajo.
    assert (
        rgb[60, 32, 1]
        > rgb[5, 32, 1]
    )


def test_prepare_index_clips_finite_values() -> None:
    patch = build_patch()

    ndmi = prepare_index(
        patch,
        "NDMI",
    )

    assert float(
        np.nanmin(ndmi)
    ) >= -1.0

    assert float(
        np.nanmax(ndmi)
    ) <= 1.0


def test_unknown_index_is_rejected() -> None:
    patch = build_patch()

    with pytest.raises(
        SpectralVisualizationError
    ):
        prepare_index(
            patch,
            "NDVI",
        )


def test_channel_statistics_ignore_nan() -> None:
    channel = np.ones(
        (
            10,
            10,
        ),
        dtype=np.float32,
    )

    channel[
        0,
        0,
    ] = np.nan

    stats = calculate_channel_statistics(
        channel
    )

    assert stats["valid_pixels"] == 99
    assert stats["total_pixels"] == 100

    assert stats[
        "valid_percentage"
    ] == pytest.approx(
        99.0
    )

    assert stats["mean"] == pytest.approx(
        1.0
    )


def test_prepare_false_color_has_expected_shape() -> None:
    patch = build_patch()

    false_color = prepare_false_color(
        patch
    )

    assert false_color.shape == (
        64,
        64,
        3,
    )
    assert false_color.dtype == np.float32
    assert float(false_color.min()) >= 0.0
    assert float(false_color.max()) <= 1.0


@pytest.mark.parametrize(
    "index_name",
    ["NDMI", "NBR"],
)
def test_prepare_index_display_has_expected_shape(
    index_name: str,
) -> None:
    patch = build_patch()

    display = prepare_index_display(
        patch,
        index_name,
    )

    assert display.shape == (
        64,
        64,
        3,
    )
    assert display.dtype == np.float32
    assert float(display.min()) >= 0.0
    assert float(display.max()) <= 1.0


def test_prepare_index_display_marks_nan_as_black() -> None:
    patch = build_patch()
    patch.data[6, 0, 0] = np.nan

    display = prepare_index_display(
        patch,
        "NDMI",
    )

    np.testing.assert_array_equal(
        display[0, 0],
        np.zeros(3, dtype=np.float32),
    )


def test_prepare_index_display_uses_viridis_extremes() -> None:
    patch = build_patch()

    display = prepare_index_display(
        patch,
        "NDMI",
    )

    np.testing.assert_allclose(
        display[0, 0],
        [0.267004, 0.004874, 0.329415],
    )
    np.testing.assert_allclose(
        display[-1, -1],
        [0.993248, 0.906157, 0.143936],
    )
