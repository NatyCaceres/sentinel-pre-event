from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.data.metadata_validator import (
    DEFAULT_SCHEMA_PATH,
    load_phase1_schema,
    validate_phase1_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "phase1"
    / "phase1_clean.csv"
)

DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "phase1"
    / "phase1_manifest.json"
)


def calculate_sha256(file_path: Path | str) -> str:
    """
    Calcula la huella SHA-256 de un archivo.

    La lectura se realiza por bloques para evitar cargar
    el archivo completo en memoria.
    """

    path = Path(file_path)

    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            sha256.update(block)

    return sha256.hexdigest().upper()


def build_phase1_manifest(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
) -> dict[str, Any]:
    """
    Construye el manifest reproducible de Fase 1.

    El dataset debe superar primero la validación.
    """

    dataset_path = Path(dataset_path)

    report = validate_phase1_file(
        csv_path=dataset_path,
        schema_path=schema_path,
    )

    if not report.is_valid:
        raise ValueError(
            "No se puede generar el manifest porque "
            "la metadata de Fase 1 no es válida."
        )

    schema = load_phase1_schema(schema_path)

    sha256 = calculate_sha256(dataset_path)

    season_min = report.stats.get("season_min")
    season_max = report.stats.get("season_max")

    dataset_id = (
        f"phase1_clean_"
        f"{season_min}_{season_max}_"
        f"{sha256[:8].lower()}"
    )

    manifest = {
        "dataset_id": dataset_id,
        "dataset_type": schema["dataset_type"],
        "schema_version": schema["schema_version"],
        "file_name": dataset_path.name,
        "sha256": sha256,
        "statistics": report.stats,
        "validation": {
            "is_valid": report.is_valid,
            "errors": report.errors,
            "warnings": report.warnings,
        },
    }

    return manifest


def save_manifest(
    manifest: dict[str, Any],
    output_path: Path | str = DEFAULT_MANIFEST_PATH,
) -> Path:
    """
    Guarda el manifest como JSON.
    """

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return path


def main() -> None:
    manifest = build_phase1_manifest()

    output_path = save_manifest(manifest)

    print()
    print("=" * 60)
    print("MANIFEST DE FASE 1")
    print("=" * 60)
    print(f"Dataset ID: {manifest['dataset_id']}")
    print(f"Archivo: {manifest['file_name']}")
    print(f"SHA-256: {manifest['sha256']}")
    print(
        "Registros: "
        f"{manifest['statistics']['rows']}"
    )
    print(
        "Temporadas: "
        f"{manifest['statistics']['season_min']}"
        "-"
        f"{manifest['statistics']['season_max']}"
    )
    print(f"Manifest guardado en: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()