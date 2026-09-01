import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.spectral.patch import (
    SpectralPatchError,
    load_spectral_patch,
)


CHANNELS = 8
HEIGHT = 64
WIDTH = 64


def create_tif(
    path,
    *,
    channels=CHANNELS,
    height=HEIGHT,
    width=WIDTH,
    data=None,
) -> None:

    if data is None:
        data = np.random.default_rng(
            42
        ).random(
            (
                channels,
                height,
                width,
            ),
            dtype=np.float32,
        )

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=channels,
        dtype="float32",
        crs="EPSG:32718",
        transform=from_origin(
            300000,
            5900000,
            10,
            10,
        ),
    ) as dataset:

        dataset.write(
            data.astype(
                np.float32
            )
        )


def test_load_valid_spectral_patch(
    tmp_path,
) -> None:

    path = tmp_path / "patch.tif"

    create_tif(path)

    patch = load_spectral_patch(
        path
    )

    assert patch.shape == (
        8,
        64,
        64,
    )

    assert patch.data.dtype == np.float32

    assert patch.channels == (
        "B2",
        "B3",
        "B4",
        "B8",
        "B11",
        "B12",
        "NDMI",
        "NBR",
    )


def test_channel_can_be_retrieved(
    tmp_path,
) -> None:

    path = tmp_path / "patch.tif"

    create_tif(path)

    patch = load_spectral_patch(
        path
    )

    ndmi = patch.channel(
        "NDMI"
    )

    assert ndmi.shape == (
        64,
        64,
    )


def test_unknown_channel_is_rejected(
    tmp_path,
) -> None:

    path = tmp_path / "patch.tif"

    create_tif(path)

    patch = load_spectral_patch(
        path
    )

    with pytest.raises(
        SpectralPatchError
    ):
        patch.channel(
            "B99"
        )


def test_wrong_number_of_channels_is_rejected(
    tmp_path,
) -> None:

    path = tmp_path / "wrong_channels.tif"

    create_tif(
        path,
        channels=7,
    )

    with pytest.raises(
        SpectralPatchError
    ):
        load_spectral_patch(
            path
        )


def test_wrong_spatial_shape_is_rejected(
    tmp_path,
) -> None:

    path = tmp_path / "wrong_shape.tif"

    create_tif(
        path,
        height=32,
        width=32,
    )

    with pytest.raises(
        SpectralPatchError
    ):
        load_spectral_patch(
            path
        )


def test_infinite_values_are_rejected(
    tmp_path,
) -> None:

    path = tmp_path / "infinite.tif"

    data = np.zeros(
        (
            8,
            64,
            64,
        ),
        dtype=np.float32,
    )

    data[
        0,
        0,
        0,
    ] = np.inf

    create_tif(
        path,
        data=data,
    )

    with pytest.raises(
        SpectralPatchError
    ):
        load_spectral_patch(
            path
        )


def test_nan_values_are_preserved(
    tmp_path,
) -> None:

    path = tmp_path / "nan.tif"

    data = np.zeros(
        (
            8,
            64,
            64,
        ),
        dtype=np.float32,
    )

    data[
        6,
        0,
        0,
    ] = np.nan

    create_tif(
        path,
        data=data,
    )

    patch = load_spectral_patch(
        path
    )

    assert np.isnan(
        patch.data[
            6,
            0,
            0,
        ]
    )