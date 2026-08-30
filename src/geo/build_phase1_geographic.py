from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.phase1 import (
    DEFAULT_PHASE1_PATH,
    Phase1Dataset,
)
from src.data.phase1_manifest import calculate_sha256
from src.geo.administrative_boundaries import (
    DEFAULT_COMMUNES_PATH,
    enrich_with_administrative_boundaries,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "phase1"
    / "phase1_geographic.csv"
)

DEFAULT_GEOGRAPHIC_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "phase1"
    / "phase1_geographic_manifest.json"
)

BOUNDARY_SOURCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "boundaries"
    / "chile_dpa_2023"
    / "source.json"
)


class GeographicBuildError(Exception):
    """
    Error asociado a la generación del artefacto
    geográfico derivado de Fase 1.
    """


def load_boundary_source() -> dict[str, Any]:
    """
    Carga la metadata de procedencia de la capa DPA.
    """

    if not BOUNDARY_SOURCE_PATH.exists():
        raise FileNotFoundError(
            "No se encontró la metadata de la DPA: "
            f"{BOUNDARY_SOURCE_PATH}"
        )

    with BOUNDARY_SOURCE_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def validate_derived_dataset(
    original: pd.DataFrame,
    enriched: pd.DataFrame,
) -> None:
    """
    Comprueba que el enriquecimiento territorial
    no haya alterado la identidad de las muestras.
    """

    if len(original) != len(enriched):
        raise GeographicBuildError(
            "El número de filas cambió durante "
            "el enriquecimiento geográfico."
        )

    if enriched["patch_id"].duplicated().any():
        raise GeographicBuildError(
            "Se generaron patch_id duplicados durante "
            "el enriquecimiento geográfico."
        )

    original_ids = set(
        original["patch_id"].astype(str)
    )

    enriched_ids = set(
        enriched["patch_id"].astype(str)
    )

    if original_ids != enriched_ids:
        raise GeographicBuildError(
            "Los patch_id del artefacto derivado "
            "no coinciden con los del dataset original."
        )


def build_phase1_geographic(
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Genera la metadata geográfica derivada de Fase 1.

    El dataset limpio original se mantiene intacto.
    """

    dataset = Phase1Dataset.load()

    original = dataset.dataframe

    enriched = enrich_with_administrative_boundaries(
        original
    )

    validate_derived_dataset(
        original=original,
        enriched=enriched,
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    enriched.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    boundary_source = load_boundary_source()

    status_counts = (
        enriched["administrative_status"]
        .value_counts()
        .to_dict()
    )

    manifest = {
        "artifact_id": (
            f"phase1_geographic_"
            f"{dataset.dataset_id}"
        ),
        "artifact_type": "phase1_geographic_metadata",
        "file_name": output_path.name,
        "sha256": calculate_sha256(output_path),
        "parent_dataset": {
            "dataset_id": dataset.dataset_id,
            "file_name": DEFAULT_PHASE1_PATH.name,
            "sha256": calculate_sha256(
                DEFAULT_PHASE1_PATH
            ),
        },
        "boundary_source": boundary_source,
        "statistics": {
            "rows": int(len(enriched)),
            "unique_patches": int(
                enriched["patch_id"].nunique()
            ),
            "assigned": int(
                status_counts.get(
                    "assigned",
                    0,
                )
            ),
            "unresolved": int(
                status_counts.get(
                    "unresolved",
                    0,
                )
            ),
            "unique_communes": int(
                enriched["COMUNA"].nunique(
                    dropna=True
                )
            ),
            "unique_regions": int(
                enriched["REGION"].nunique(
                    dropna=True
                )
            ),
        },
    }

    return enriched, manifest


def save_manifest(
    manifest: dict[str, Any],
    output_path: Path | str = DEFAULT_GEOGRAPHIC_MANIFEST_PATH,
) -> Path:
    """
    Guarda el manifest del artefacto geográfico.
    """

    output_path = Path(output_path)

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
    enriched, manifest = build_phase1_geographic()

    manifest_path = save_manifest(
        manifest
    )

    stats = manifest["statistics"]

    print()
    print("=" * 60)
    print("ARTEFACTO GEOGRÁFICO — FASE 1")
    print("=" * 60)
    print(
        f"Dataset origen: "
        f"{manifest['parent_dataset']['dataset_id']}"
    )
    print(
        f"Registros: {stats['rows']}"
    )
    print(
        f"Asignados: {stats['assigned']}"
    )
    print(
        f"Sin asignación: {stats['unresolved']}"
    )
    print(
        f"Comunas representadas: "
        f"{stats['unique_communes']}"
    )
    print(
        f"Regiones representadas: "
        f"{stats['unique_regions']}"
    )
    print(
        f"SHA-256: {manifest['sha256']}"
    )
    print(
        f"Manifest: {manifest_path}"
    )
    print("=" * 60)

    unresolved = enriched[
        enriched["administrative_status"]
        == "unresolved"
    ]

    if not unresolved.empty:
        print()
        print("REGISTROS SIN ASIGNACIÓN ADMINISTRATIVA")
        print("-" * 60)

        print(
            unresolved[
                [
                    "patch_id",
                    "lat_centro",
                    "lon_centro",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()