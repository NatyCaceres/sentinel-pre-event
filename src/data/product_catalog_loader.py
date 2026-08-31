from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.product_catalog import (
    DEFAULT_CATALOG_MANIFEST_PATH,
    DEFAULT_CATALOG_PATH,
    calculate_sha256,
)


class ProductCatalogLoadError(Exception):
    """
    Error asociado a la carga del catálogo
    utilizado en tiempo de ejecución por el producto.
    """


def load_catalog_manifest(
    manifest_path: Path | str = DEFAULT_CATALOG_MANIFEST_PATH,
) -> dict:
    """
    Carga el manifest del catálogo del producto.
    """

    path = Path(manifest_path)

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el manifest del catálogo: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_product_catalog(
    catalog_path: Path | str = DEFAULT_CATALOG_PATH,
    manifest_path: Path | str = DEFAULT_CATALOG_MANIFEST_PATH,
    *,
    verify_integrity: bool = True,
) -> pd.DataFrame:
    """
    Carga y valida el catálogo que consume la aplicación.

    El catálogo representa únicamente los parches
    Sentinel-2 disponibles para el producto.
    """

    catalog_path = Path(catalog_path)

    if not catalog_path.exists():
        raise FileNotFoundError(
            f"No se encontró el catálogo del producto: {catalog_path}"
        )

    manifest = load_catalog_manifest(
        manifest_path
    )

    if verify_integrity:

        current_hash = calculate_sha256(
            catalog_path
        )

        expected_hash = manifest["sha256"]

        if current_hash != expected_hash:
            raise ProductCatalogLoadError(
                "El SHA-256 del catálogo no coincide "
                "con su manifest."
            )

    catalog = pd.read_csv(
        catalog_path
    )

    required_columns = {
        "patch_id",
        "group_id",
        "etiqueta",
        "lat_centro",
        "lon_centro",
        "season_year",
        "fecha_referencia",
        "fecha_objetivo_pre_evento",
        "image_id",
        "fecha_imagen_seleccionada",
        "tif_filename",
        "image_available",
        "map_eligible",
        "COMUNA",
        "PROVINCIA",
        "REGION",
        "administrative_status",
        "n_patches_group_available",
        "es_multipatch_available",
    }

    missing_columns = (
        required_columns - set(catalog.columns)
    )

    if missing_columns:
        raise ProductCatalogLoadError(
            "El catálogo no cumple la estructura "
            "esperada. Faltan columnas: "
            f"{sorted(missing_columns)}"
        )

    stats = manifest["statistics"]

    expected_rows = int(
        stats["rows"]
    )

    if len(catalog) != expected_rows:
        raise ProductCatalogLoadError(
            "La cantidad de registros del catálogo "
            "no coincide con su manifest."
        )

    expected_patches = int(
        stats["unique_patches"]
    )

    if (
        catalog["patch_id"].nunique()
        != expected_patches
    ):
        raise ProductCatalogLoadError(
            "La cantidad de patch_id únicos no coincide "
            "con su manifest."
        )

    if catalog["patch_id"].duplicated().any():
        raise ProductCatalogLoadError(
            "El catálogo contiene patch_id duplicados."
        )

    if not (
        catalog["image_available"]
        .fillna(False)
        .astype(bool)
        .all()
    ):
        raise ProductCatalogLoadError(
            "El catálogo contiene registros sin "
            "imagen Sentinel-2 disponible."
        )

    return catalog