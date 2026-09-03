import hashlib
import json

import pandas as pd
import pytest

from src.data.product_catalog_loader import (
    ProductCatalogLoadError,
    load_product_catalog,
    select_final_test_catalog,
)


def calculate_test_sha256(path) -> str:
    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        sha256.update(file.read())

    return sha256.hexdigest().upper()


def build_valid_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patch_id": [
                "P_001",
                "P_002",
            ],
            "group_id": [
                "EV_001",
                "NEG_001",
            ],
            "subsplit": [
                "test",
                "train",
            ],
            "etiqueta": [
                1,
                0,
            ],
            "lat_centro": [
                -37.0,
                -38.0,
            ],
            "lon_centro": [
                -72.0,
                -73.0,
            ],
            "season_year": [
                2025,
                2025,
            ],
            "fecha_referencia": [
                "2026-01-10",
                "2026-02-10",
            ],
            "fecha_objetivo_pre_evento": [
                "2025-10-10",
                "2025-11-10",
            ],
            "image_id": [
                "IMG_001",
                "IMG_002",
            ],
            "fecha_imagen_seleccionada": [
                "2025-10-12",
                "2025-11-12",
            ],
            "tif_filename": [
                "P_001.tif",
                "P_002.tif",
            ],
            "image_available": [
                True,
                True,
            ],
            "map_eligible": [
                True,
                True,
            ],
            "COMUNA": [
                "Comuna A",
                "Comuna B",
            ],
            "PROVINCIA": [
                "Provincia A",
                "Provincia B",
            ],
            "REGION": [
                "Región A",
                "Región B",
            ],
            "administrative_status": [
                "assigned",
                "assigned",
            ],
            "n_patches_group_available": [
                1,
                1,
            ],
            "es_multipatch_available": [
                False,
                False,
            ],
        }
    )


def create_test_files(tmp_path):
    catalog_path = (
        tmp_path
        / "product_catalog.csv"
    )

    manifest_path = (
        tmp_path
        / "product_catalog_manifest.json"
    )

    catalog = build_valid_catalog()

    catalog.to_csv(
        catalog_path,
        index=False,
    )

    sha256 = calculate_test_sha256(
        catalog_path
    )

    manifest = {
        "sha256": sha256,
        "statistics": {
            "rows": 2,
            "unique_patches": 2,
        },
    }

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
        )

    return (
        catalog_path,
        manifest_path,
    )


def test_load_valid_catalog(
    tmp_path,
) -> None:
    catalog_path, manifest_path = (
        create_test_files(tmp_path)
    )

    result = load_product_catalog(
        catalog_path,
        manifest_path,
    )

    assert len(result) == 2
    assert result["patch_id"].nunique() == 2
    assert result["image_available"].all()


def test_modified_catalog_is_rejected(
    tmp_path,
) -> None:
    catalog_path, manifest_path = (
        create_test_files(tmp_path)
    )

    catalog = pd.read_csv(
        catalog_path
    )

    catalog.loc[
        0,
        "patch_id",
    ] = "ALTERADO"

    catalog.to_csv(
        catalog_path,
        index=False,
    )

    with pytest.raises(
        ProductCatalogLoadError
    ):
        load_product_catalog(
            catalog_path,
            manifest_path,
        )


def test_catalog_without_required_column_is_rejected(
    tmp_path,
) -> None:
    catalog_path, manifest_path = (
        create_test_files(tmp_path)
    )

    catalog = pd.read_csv(
        catalog_path
    ).drop(
        columns=["image_id"]
    )

    catalog.to_csv(
        catalog_path,
        index=False,
    )

    # Desactivamos SHA para comprobar específicamente
    # el contrato estructural.
    with pytest.raises(
        ProductCatalogLoadError
    ):
        load_product_catalog(
            catalog_path,
            manifest_path,
            verify_integrity=False,
        )


def test_catalog_without_available_image_is_rejected(
    tmp_path,
) -> None:
    catalog_path, manifest_path = (
        create_test_files(tmp_path)
    )

    catalog = pd.read_csv(
        catalog_path
    )

    catalog.loc[
        0,
        "image_available",
    ] = False

    catalog.to_csv(
        catalog_path,
        index=False,
    )

    with pytest.raises(
        ProductCatalogLoadError
    ):
        load_product_catalog(
            catalog_path,
            manifest_path,
            verify_integrity=False,
        )


def test_select_final_test_catalog_keeps_only_test_rows() -> None:
    catalog = build_valid_catalog()

    result = select_final_test_catalog(
        catalog
    )

    assert result["patch_id"].tolist() == [
        "P_001"
    ]
    assert result["subsplit"].tolist() == [
        "test"
    ]
    assert len(catalog) == 2


def test_select_final_test_catalog_rejects_missing_partition() -> None:
    catalog = build_valid_catalog()
    catalog["subsplit"] = "train"

    with pytest.raises(
        ProductCatalogLoadError
    ):
        select_final_test_catalog(
            catalog
        )
