from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.phase1_manifest import calculate_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_GEOGRAPHIC_PATH = (
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


class GeographicArtifactError(Exception):
    """
    Error asociado al artefacto geográfico de Fase 1.
    """


def load_geographic_manifest(
    manifest_path: Path | str = DEFAULT_GEOGRAPHIC_MANIFEST_PATH,
) -> dict:
    """
    Carga el manifest del artefacto geográfico.
    """

    path = Path(manifest_path)

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el manifest geográfico: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_phase1_geographic(
    csv_path: Path | str = DEFAULT_GEOGRAPHIC_PATH,
    manifest_path: Path | str = DEFAULT_GEOGRAPHIC_MANIFEST_PATH,
    verify_integrity: bool = True,
) -> pd.DataFrame:
    """
    Carga el artefacto geográfico derivado de Fase 1.

    Antes de devolver los datos comprueba:
    - existencia del archivo;
    - estructura mínima;
    - número de registros;
    - identidad de patches;
    - SHA-256, si se solicita.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontró el artefacto geográfico: {csv_path}"
        )

    manifest = load_geographic_manifest(
        manifest_path
    )

    if verify_integrity:
        current_hash = calculate_sha256(
            csv_path
        )

        expected_hash = manifest["sha256"]

        if current_hash != expected_hash:
            raise GeographicArtifactError(
                "El SHA-256 del artefacto geográfico "
                "no coincide con el manifest."
            )

    dataframe = pd.read_csv(
        csv_path
    )

    required_columns = {
        "patch_id",
        "group_id",
        "etiqueta",
        "lat_centro",
        "lon_centro",
        "season_year",
        "fecha_referencia",
        "COMUNA",
        "PROVINCIA",
        "REGION",
        "administrative_status",
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise GeographicArtifactError(
            "Faltan columnas obligatorias en el "
            "artefacto geográfico: "
            f"{sorted(missing_columns)}"
        )

    expected_rows = int(
        manifest["statistics"]["rows"]
    )

    if len(dataframe) != expected_rows:
        raise GeographicArtifactError(
            "El número de registros no coincide "
            "con el manifest geográfico."
        )

    expected_patches = int(
        manifest["statistics"]["unique_patches"]
    )

    if (
        dataframe["patch_id"].nunique()
        != expected_patches
    ):
        raise GeographicArtifactError(
            "La cantidad de patch_id únicos no coincide "
            "con el manifest geográfico."
        )

    return dataframe