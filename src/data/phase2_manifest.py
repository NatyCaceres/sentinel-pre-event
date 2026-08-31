from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.phase2_validator import (
    DEFAULT_PHASE2_PATH,
    load_schema,
    validate_phase2_metadata,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "phase2"
    / "phase2_manifest.json"
)


class Phase2ManifestError(Exception):
    """
    Error asociado a la generación del manifest de Fase 2.
    """


def calculate_sha256(
    path: Path | str,
) -> str:
    """
    Calcula la huella SHA-256 de un archivo.
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


def build_phase2_manifest(
    metadata_path: Path | str = DEFAULT_PHASE2_PATH,
) -> dict[str, Any]:
    """
    Valida la metadata y genera su manifest reproducible.
    """

    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No se encontró la metadata de Fase 2: "
            f"{metadata_path}"
        )

    dataframe = pd.read_csv(
        metadata_path
    )

    schema = load_schema()

    errors = validate_phase2_metadata(
        dataframe,
        schema,
    )

    if errors:
        raise Phase2ManifestError(
            "No se puede generar el manifest porque "
            "la metadata de Fase 2 es inválida: "
            + "; ".join(errors)
        )

    sha256 = calculate_sha256(
        metadata_path
    )

    seasons = sorted(
        dataframe["season_year"]
        .astype(int)
        .unique()
        .tolist()
    )

    class_counts = (
        dataframe["etiqueta"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    split_counts = (
        dataframe["split"]
        .value_counts()
        .to_dict()
    )

    subsplit_counts = (
        dataframe["subsplit"]
        .value_counts()
        .to_dict()
    )

    dataset_id = (
        f"phase2_sentinel8ch_"
        f"{min(seasons)}_{max(seasons)}_"
        f"{sha256[:8].lower()}"
    )

    return {
        "dataset_id": dataset_id,
        "artifact_type": "phase2_modeling_metadata",
        "file_name": metadata_path.name,
        "sha256": sha256,
        "statistics": {
            "rows": int(len(dataframe)),
            "unique_patches": int(
                dataframe["patch_id"].nunique()
            ),
            "unique_groups": int(
                dataframe["group_id"].nunique()
            ),
            "class_counts": {
                str(key): int(value)
                for key, value in class_counts.items()
            },
            "split_counts": {
                str(key): int(value)
                for key, value in split_counts.items()
            },
            "subsplit_counts": {
                str(key): int(value)
                for key, value in subsplit_counts.items()
            },
            "seasons": seasons,
            "available_tifs": int(
                dataframe["existe_tif"].sum()
            ),
        },
        "data_contract": {
            "channels": [
                "B2",
                "B3",
                "B4",
                "B8",
                "B11",
                "B12",
                "NDMI",
                "NBR",
            ],
            "expected_channels": 8,
            "patch_id_unique": True,
            "availability_field": "existe_tif",
        },
    }


def save_phase2_manifest(
    manifest: dict[str, Any],
    output_path: Path | str = DEFAULT_MANIFEST_PATH,
) -> Path:
    """
    Guarda el manifest en disco.
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

    manifest = build_phase2_manifest()

    path = save_phase2_manifest(
        manifest
    )

    stats = manifest["statistics"]

    print()
    print("=" * 60)
    print("MANIFEST METADATA — FASE 2")
    print("=" * 60)

    print(
        f"Dataset ID: {manifest['dataset_id']}"
    )

    print(
        f"Registros: {stats['rows']}"
    )

    print(
        f"Patches únicos: {stats['unique_patches']}"
    )

    print(
        f"Grupos únicos: {stats['unique_groups']}"
    )

    print(
        f"TIFF disponibles: {stats['available_tifs']}"
    )

    print(
        f"Temporadas: {stats['seasons']}"
    )

    print(
        f"SHA-256: {manifest['sha256']}"
    )

    print(
        f"Manifest: {path}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()