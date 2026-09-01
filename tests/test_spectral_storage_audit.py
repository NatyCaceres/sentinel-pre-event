import json

import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine

from src.spectral.patch import (
    DEFAULT_CONTRACT_PATH,
)
from src.spectral.storage_audit import (
    SpectralStorageAuditError,
    audit_is_valid,
    audit_spectral_storage,
    build_storage_manifest,
    save_storage_manifest,
    validate_stored_tif,
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
    dtype="float32",
    data=None,
) -> None:

    if data is None:
        data = np.zeros(
            (
                channels,
                height,
                width,
            ),
            dtype=dtype,
        )

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=channels,
        dtype=dtype,
        crs="EPSG:32718",
        transform=Affine(
            10,
            0,
            300000,
            0,
            -10,
            5900000,
        ),
    ) as dataset:
        dataset.write(
            data.astype(dtype)
        )


def build_catalog(
    *filenames: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tif_filename": list(
                filenames
            )
        }
    )


def test_valid_storage_audit(
    tmp_path,
) -> None:
    create_tif(
        tmp_path / "patch.tif"
    )

    report = audit_spectral_storage(
        catalog=build_catalog(
            "patch.tif"
        ),
        patch_directory=tmp_path,
    )

    assert report["is_valid"]
    assert audit_is_valid(report)
    assert report["summary"] == {
        "expected_tiffs": 1,
        "expected_unique_tiffs": 1,
        "tiffs_found": 1,
        "matched_tiffs": 1,
        "missing_tiffs": 0,
        "extra_tiffs": 0,
        "duplicate_names": 0,
        "invalid_catalog_entries": 0,
        "physically_valid_tiffs": 1,
        "physically_invalid_tiffs": 0,
    }


def test_missing_and_extra_tiffs_are_reported(
    tmp_path,
) -> None:
    create_tif(
        tmp_path / "extra.tif"
    )

    report = audit_spectral_storage(
        catalog=build_catalog(
            "missing.tif"
        ),
        patch_directory=tmp_path,
    )

    assert not report["is_valid"]
    assert report["missing"] == [
        "missing.tif"
    ]
    assert report["extras"] == [
        "extra.tif"
    ]


def test_duplicate_catalog_names_are_reported(
    tmp_path,
) -> None:
    create_tif(
        tmp_path / "patch.tif"
    )

    report = audit_spectral_storage(
        catalog=build_catalog(
            "patch.tif",
            "patch.tif",
        ),
        patch_directory=tmp_path,
    )

    assert not report["is_valid"]
    assert report["duplicates"] == [
        {
            "filename": "patch.tif",
            "count": 2,
        }
    ]


def test_case_only_name_collision_is_reported(
    tmp_path,
) -> None:
    create_tif(
        tmp_path / "Patch.tif"
    )

    report = audit_spectral_storage(
        catalog=build_catalog(
            "Patch.tif",
            "patch.tif",
        ),
        patch_directory=tmp_path,
    )

    assert not report["is_valid"]
    assert report["duplicates"] == [
        {
            "filename": "Patch.tif",
            "count": 2,
            "variants": [
                "Patch.tif",
                "patch.tif",
            ],
        }
    ]


def test_invalid_catalog_filename_is_reported(
    tmp_path,
) -> None:
    report = audit_spectral_storage(
        catalog=build_catalog(
            "folder/patch.tif"
        ),
        patch_directory=tmp_path,
    )

    assert not report["is_valid"]
    assert (
        report["summary"][
            "invalid_catalog_entries"
        ]
        == 1
    )


def test_corrupt_tif_is_reported(
    tmp_path,
) -> None:
    path = tmp_path / "corrupt.tif"
    path.write_bytes(b"not a geotiff")

    result = validate_stored_tif(
        path
    )

    assert not result["valid"]
    assert "No fue posible abrir" in (
        result["errors"][0]
    )


@pytest.mark.parametrize(
    (
        "creation_options",
        "expected_error",
    ),
    [
        (
            {"channels": 7},
            "Cantidad de canales inválida",
        ),
        (
            {"height": 32},
            "Dimensiones espaciales inválidas",
        ),
        (
            {"dtype": "uint16"},
            "dtype físico inválido",
        ),
    ],
)
def test_invalid_physical_contract_is_reported(
    tmp_path,
    creation_options,
    expected_error,
) -> None:
    path = tmp_path / "invalid.tif"
    create_tif(
        path,
        **creation_options,
    )

    result = validate_stored_tif(
        path
    )

    assert not result["valid"]
    assert any(
        expected_error in error
        for error in result["errors"]
    )


def test_infinite_values_are_reported(
    tmp_path,
) -> None:
    path = tmp_path / "infinite.tif"
    data = np.zeros(
        (
            CHANNELS,
            HEIGHT,
            WIDTH,
        ),
        dtype=np.float32,
    )
    data[0, 0, 0] = np.inf
    create_tif(
        path,
        data=data,
    )

    result = validate_stored_tif(
        path
    )

    assert not result["valid"]
    assert any(
        "infinitos" in error
        for error in result["errors"]
    )


def test_nan_values_are_accepted(
    tmp_path,
) -> None:
    path = tmp_path / "nan.tif"
    data = np.zeros(
        (
            CHANNELS,
            HEIGHT,
            WIDTH,
        ),
        dtype=np.float32,
    )
    data[6, 0, 0] = np.nan
    create_tif(
        path,
        data=data,
    )

    result = validate_stored_tif(
        path
    )

    assert result["valid"]


def test_manifest_is_deterministic_and_portable(
    tmp_path,
) -> None:
    tif_name = "patch.tif"
    create_tif(
        tmp_path / tif_name
    )
    catalog = build_catalog(
        tif_name
    )
    catalog_path = tmp_path / "catalog.csv"
    catalog.to_csv(
        catalog_path,
        index=False,
    )

    report = audit_spectral_storage(
        catalog=catalog,
        patch_directory=tmp_path,
    )

    first = build_storage_manifest(
        report,
        catalog_path=catalog_path,
        contract_path=DEFAULT_CONTRACT_PATH,
    )
    second = build_storage_manifest(
        report,
        catalog_path=catalog_path,
        contract_path=DEFAULT_CONTRACT_PATH,
    )

    assert first == second
    assert first["files"][0][
        "filename"
    ] == tif_name
    assert "patch_directory" not in first
    assert "generated_at" not in first

    output_path = (
        tmp_path
        / "spectral_storage_manifest.json"
    )
    save_storage_manifest(
        first,
        output_path,
    )

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        saved = json.load(file)

    assert saved == first


def test_invalid_audit_cannot_build_manifest(
    tmp_path,
) -> None:
    catalog = build_catalog(
        "missing.tif"
    )
    catalog_path = tmp_path / "catalog.csv"
    catalog.to_csv(
        catalog_path,
        index=False,
    )
    report = audit_spectral_storage(
        catalog=catalog,
        patch_directory=tmp_path,
    )

    with pytest.raises(
        SpectralStorageAuditError
    ):
        build_storage_manifest(
            report,
            catalog_path=catalog_path,
        )
