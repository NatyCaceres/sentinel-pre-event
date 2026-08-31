from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.phase1_geographic import (
    DEFAULT_GEOGRAPHIC_MANIFEST_PATH,
    load_phase1_geographic,
)
from src.data.phase12_audit import (
    audit_is_compatible,
    audit_phase1_phase2,
    load_phase2_metadata,
)
from src.data.phase2_manifest import (
    DEFAULT_MANIFEST_PATH as DEFAULT_PHASE2_MANIFEST_PATH,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CATALOG_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "product"
    / "product_catalog.csv"
)

DEFAULT_CATALOG_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "product"
    / "product_catalog_manifest.json"
)


class ProductCatalogError(Exception):
    """
    Error asociado a la construcción del catálogo
    principal del producto.
    """


def calculate_sha256(
    path: Path | str,
) -> str:
    """
    Calcula SHA-256 de un archivo.
    """

    path = Path(path)

    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            sha256.update(block)

    return sha256.hexdigest().upper()


def load_json(
    path: Path | str,
) -> dict[str, Any]:
    """
    Carga un archivo JSON.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def build_product_catalog(
    phase1: pd.DataFrame,
    phase2: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye el catálogo que consumirá el producto.

    Fase 2 debe ser un subconjunto consistente de Fase 1.
    """

    audit = audit_phase1_phase2(
        phase1,
        phase2,
    )

    if not audit_is_compatible(audit):
        raise ProductCatalogError(
            "Fase 1 y Fase 2 no son compatibles. "
            "No se puede generar el catálogo."
        )

    # --------------------------------------------------------
    # CONTEXTO QUE TOMAMOS DE FASE 1
    # --------------------------------------------------------

    phase1_columns = [
        "patch_id",
        "COMUNA",
        "PROVINCIA",
        "REGION",
        "administrative_status",
    ]

    geographic_context = phase1[
        phase1_columns
    ].copy()

    # --------------------------------------------------------
    # CAMPOS DEL PRODUCTO QUE TOMAMOS DE FASE 2
    # --------------------------------------------------------

    phase2_columns = [
        "patch_id",
        "group_id",
        "etiqueta",
        "lat_centro",
        "lon_centro",
        "season_year",
        "banda_lat",
        "fecha_referencia",
        "fecha_objetivo_pre_evento",
        "gee_inicio",
        "gee_fin",
        "spatial_block",
        "split",
        "event_weight",
        "image_id",
        "fecha_imagen_seleccionada",
        "dias_diferencia_objetivo",
        "pct_pixeles_validos_patch",
        "cloudy_pixel_percentage",
        "tipo_seleccion",
        "n_assets_candidatos",
        "n_imagenes_candidatas",
        "n_imagenes_validas",
        "archivo_model8_esperado",
        "ruta_model_8ch",
        "existe_tif",
        "subsplit",
        "event_weight_model",
    ]

    missing_phase2_columns = (
        set(phase2_columns)
        - set(phase2.columns)
    )

    if missing_phase2_columns:
        raise ProductCatalogError(
            "Faltan columnas requeridas de Fase 2: "
            f"{sorted(missing_phase2_columns)}"
        )

    catalog = phase2[
        phase2_columns
    ].copy()

    catalog = catalog.merge(
        geographic_context,
        on="patch_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # DISPONIBILIDAD PARA EL PRODUCTO
    # --------------------------------------------------------

    catalog["image_available"] = (
        catalog["existe_tif"]
        .fillna(False)
        .astype(bool)
    )

    catalog["map_eligible"] = (
        catalog["image_available"]
        & catalog[
            "administrative_status"
        ].eq("assigned")
    )

    # Nombre portable del archivo.
    # No utilizamos /content/gdrive/... como ruta operacional.
    catalog["tif_filename"] = (
        catalog["archivo_model8_esperado"]
        .astype(str)
    )

    # --------------------------------------------------------
    # MULTIPATCH RECALCULADO SOBRE LAS IMÁGENES DISPONIBLES
    # --------------------------------------------------------

    available_group_sizes = (
        catalog[
            catalog["image_available"]
        ]
        .groupby("group_id")["patch_id"]
        .transform("size")
    )

    catalog[
        "n_patches_group_available"
    ] = available_group_sizes.astype(int)

    catalog[
        "es_multipatch_available"
    ] = (
        catalog[
            "n_patches_group_available"
        ] > 1
    )

    # --------------------------------------------------------
    # ORDEN Y VALIDACIONES FINALES
    # --------------------------------------------------------

    if catalog["patch_id"].duplicated().any():
        raise ProductCatalogError(
            "El catálogo contiene patch_id duplicados."
        )

    if len(catalog) != len(phase2):
        raise ProductCatalogError(
            "El catálogo no conserva el número "
            "de registros de Fase 2."
        )

    if catalog[
        "administrative_status"
    ].isna().any():
        raise ProductCatalogError(
            "Hay patches de Fase 2 sin correspondencia "
            "con la metadata geográfica de Fase 1."
        )

    return catalog.reset_index(
        drop=True
    )


def build_catalog_manifest(
    catalog: pd.DataFrame,
    catalog_path: Path,
) -> dict[str, Any]:
    """
    Construye el manifest del catálogo del producto.
    """

    phase1_manifest = load_json(
        DEFAULT_GEOGRAPHIC_MANIFEST_PATH
    )

    phase2_manifest = load_json(
        DEFAULT_PHASE2_MANIFEST_PATH
    )

    sha256 = calculate_sha256(
        catalog_path
    )

    group_sizes = (
        catalog.groupby(
            "group_id"
        )["patch_id"]
        .size()
    )

    return {
        "catalog_id": (
            f"product_catalog_"
            f"{sha256[:8].lower()}"
        ),
        "artifact_type": "product_catalog",
        "file_name": catalog_path.name,
        "sha256": sha256,
        "parents": {
            "phase1_geographic": {
                "artifact_id": (
                    phase1_manifest.get(
                        "artifact_id"
                    )
                ),
                "sha256": phase1_manifest[
                    "sha256"
                ],
            },
            "phase2": {
                "dataset_id": phase2_manifest[
                    "dataset_id"
                ],
                "sha256": phase2_manifest[
                    "sha256"
                ],
            },
        },
        "statistics": {
            "rows": int(len(catalog)),
            "unique_patches": int(
                catalog["patch_id"].nunique()
            ),
            "unique_groups": int(
                catalog["group_id"].nunique()
            ),
            "image_available": int(
                catalog[
                    "image_available"
                ].sum()
            ),
            "map_eligible": int(
                catalog[
                    "map_eligible"
                ].sum()
            ),
            "administrative_unresolved": int(
                (
                    catalog[
                        "administrative_status"
                    ] == "unresolved"
                ).sum()
            ),
            "multipatch_groups_available": int(
                (group_sizes > 1).sum()
            ),
            "class_counts": {
                str(key): int(value)
                for key, value
                in catalog[
                    "etiqueta"
                ]
                .value_counts()
                .sort_index()
                .items()
            },
        },
    }


def save_catalog_manifest(
    manifest: dict[str, Any],
    output_path: Path | str = DEFAULT_CATALOG_MANIFEST_PATH,
) -> Path:
    """
    Guarda el manifest del catálogo.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return output_path


def main() -> None:
    phase1 = load_phase1_geographic()
    phase2 = load_phase2_metadata()

    catalog = build_product_catalog(
        phase1,
        phase2,
    )

    DEFAULT_CATALOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    catalog.to_csv(
        DEFAULT_CATALOG_PATH,
        index=False,
        encoding="utf-8",
    )

    manifest = build_catalog_manifest(
        catalog,
        DEFAULT_CATALOG_PATH,
    )

    manifest_path = save_catalog_manifest(
        manifest
    )

    stats = manifest["statistics"]

    print()
    print("=" * 65)
    print("CATÁLOGO DEL PRODUCTO")
    print("=" * 65)

    print(
        f"Catalog ID: {manifest['catalog_id']}"
    )

    print(
        f"Registros: {stats['rows']}"
    )

    print(
        f"Patches únicos: "
        f"{stats['unique_patches']}"
    )

    print(
        f"Grupos únicos: "
        f"{stats['unique_groups']}"
    )

    print(
        f"Con imagen disponible: "
        f"{stats['image_available']}"
    )

    print(
        f"Elegibles para mapa: "
        f"{stats['map_eligible']}"
    )

    print(
        "Sin asignación administrativa: "
        f"{stats['administrative_unresolved']}"
    )

    print(
        "Grupos multipatch disponibles: "
        f"{stats['multipatch_groups_available']}"
    )

    print(
        f"SHA-256: {manifest['sha256']}"
    )

    print(
        f"Manifest: {manifest_path}"
    )

    print("=" * 65)


if __name__ == "__main__":
    main()