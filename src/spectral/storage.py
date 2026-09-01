from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PATCH_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "patches"
    / "sentinel_8ch"
)

PATCH_DIRECTORY_ENV = "SENTINEL_PATCH_DIR"


class SpectralStorageError(Exception):
    """
    Error asociado al almacenamiento o resolución
    de GeoTIFF Sentinel-2.
    """


def get_patch_directory() -> Path:
    """
    Obtiene el directorio operacional de GeoTIFF.

    Prioridad:
    1. Variable de entorno SENTINEL_PATCH_DIR.
    2. data/patches/sentinel_8ch del proyecto.
    """

    configured_path = os.getenv(
        PATCH_DIRECTORY_ENV
    )

    if configured_path:

        return Path(
            configured_path
        ).expanduser().resolve()

    return DEFAULT_PATCH_DIRECTORY.resolve()


def resolve_tif_path(
    tif_filename: str,
    *,
    require_exists: bool = True,
) -> Path:
    """
    Convierte un nombre portable de TIFF
    en una ruta operacional local.
    """

    if not tif_filename:
        raise SpectralStorageError(
            "El nombre del GeoTIFF está vacío."
        )

    filename = Path(
        str(tif_filename)
    ).name

    if filename != str(tif_filename):
        raise SpectralStorageError(
            "tif_filename debe contener únicamente "
            "el nombre del archivo, no una ruta."
        )

    if not filename.lower().endswith(
        (".tif", ".tiff")
    ):
        raise SpectralStorageError(
            f"Extensión GeoTIFF inválida: {filename}"
        )

    path = (
        get_patch_directory()
        / filename
    )

    if (
        require_exists
        and not path.exists()
    ):
        raise FileNotFoundError(
            f"No se encontró el GeoTIFF: {path}"
        )

    return path


def resolve_patch_tif(
    patch: pd.Series,
    *,
    require_exists: bool = True,
) -> Path:
    """
    Resuelve el GeoTIFF asociado a una fila
    del catálogo del producto.
    """

    if "tif_filename" not in patch.index:
        raise SpectralStorageError(
            "El registro no contiene tif_filename."
        )

    return resolve_tif_path(
        patch["tif_filename"],
        require_exists=require_exists,
    )