from __future__ import annotations

from datetime import date, datetime

import pandas as pd


class ExplorerFilterError(Exception):
    """
    Error asociado a los filtros del explorador geográfico.
    """


def apply_explorer_filters(
    points: pd.DataFrame,
    *,
    season_year: int | None = None,
    region: str | None = None,
    label: int | None = None,
    reference_start: date | datetime | pd.Timestamp | None = None,
    reference_end: date | datetime | pd.Timestamp | None = None,
    multipatch_only: bool = False,
) -> pd.DataFrame:
    """
    Aplica los filtros disponibles en el explorador geográfico.

    Los datos originales no son modificados.

    Parameters
    ----------
    points:
        DataFrame preparado para el mapa.

    season_year:
        Año interno de temporada. Por ejemplo, 2025 representa
        visualmente la temporada 2025–2026.

    region:
        Nombre de la región administrativa.

    label:
        0 = sin detección cercana.
        1 = con detección posterior.

    reference_start / reference_end:
        Intervalo de fecha de referencia, inclusivo.

    multipatch_only:
        Si es True, conserva únicamente muestras pertenecientes
        a grupos con más de un parche.
    """

    required_columns = {
        "season_year",
        "REGION",
        "etiqueta",
        "fecha_referencia",
        "es_multipatch",
    }

    missing_columns = (
        required_columns - set(points.columns)
    )

    if missing_columns:
        raise ExplorerFilterError(
            "No se pueden aplicar los filtros. "
            f"Faltan columnas: {sorted(missing_columns)}"
        )

    if label not in {None, 0, 1}:
        raise ExplorerFilterError(
            f"Etiqueta inválida para el filtro: {label}"
        )

    result = points.copy()

    if season_year is not None:
        result = result[
            result["season_year"] == int(season_year)
        ]

    if region is not None:
        result = result[
            result["REGION"] == region
        ]

    if label is not None:
        result = result[
            result["etiqueta"] == label
        ]

    reference_dates = pd.to_datetime(
        result["fecha_referencia"],
        errors="coerce",
    )

    if reference_start is not None:
        start = pd.Timestamp(reference_start)

        result = result[
            reference_dates >= start
        ]

        reference_dates = pd.to_datetime(
            result["fecha_referencia"],
            errors="coerce",
        )

    if reference_end is not None:
        end = pd.Timestamp(reference_end)

        result = result[
            reference_dates <= end
        ]

    if multipatch_only:
        result = result[
            result["es_multipatch"]
        ]

    return result.reset_index(drop=True)