from __future__ import annotations

from typing import Any

import pandas as pd


def format_season_label(season_year: int) -> str:
    """
    Convierte el identificador interno de una temporada
    en una etiqueta legible para la interfaz.

    Ejemplo
    -------
    2025 -> "2025–2026"
    """

    start_year = int(season_year)
    end_year = start_year + 1

    return f"{start_year}–{end_year}"


def format_class_label(label: int) -> str:
    """
    Convierte la etiqueta numérica en una descripción
    adecuada para la interfaz.
    """

    labels = {
        0: "Sin detección cercana",
        1: "Con detección posterior",
    }

    try:
        return labels[int(label)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Etiqueta de clase no válida: {label}"
        ) from exc


def build_map_points(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepara las muestras de Fase 1 para su representación
    en el mapa interactivo.

    No modifica el DataFrame recibido.
    """

    required_columns = {
        "patch_id",
        "group_id",
        "etiqueta",
        "lat_centro",
        "lon_centro",
        "season_year",
        "banda_lat",
        "fecha_referencia",
        "split",
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "No se pueden generar puntos geográficos. "
            "Faltan columnas: "
            f"{sorted(missing_columns)}"
        )

    points = dataframe.copy()

    group_sizes = (
        points
        .groupby("group_id")
        .size()
        .rename("n_patches_grupo")
    )

    points = points.join(
        group_sizes,
        on="group_id",
    )

    points["es_multipatch"] = (
        points["n_patches_grupo"] > 1
    )

    points["temporada"] = (
        pd.to_numeric(
            points["season_year"],
            errors="raise",
        )
        .astype(int)
        .map(format_season_label)
    )

    points["clase"] = (
        pd.to_numeric(
            points["etiqueta"],
            errors="raise",
        )
        .astype(int)
        .map(format_class_label)
    )

    points["fecha_referencia"] = (
        pd.to_datetime(
            points["fecha_referencia"],
            errors="raise",
        )
        .dt.date
    )

    points = points.rename(
        columns={
            "lat_centro": "lat",
            "lon_centro": "lon",
            "banda_lat": "zona",
        }
    )

    output_columns = [
        "patch_id",
        "group_id",
        "lat",
        "lon",
        "temporada",
        "season_year",
        "etiqueta",
        "clase",
        "fecha_referencia",
        "zona",
        "split",
        "n_patches_grupo",
        "es_multipatch",
    ]

    return (
        points[output_columns]
        .reset_index(drop=True)
        .copy()
    )


def get_map_summary(
    map_points: pd.DataFrame,
) -> dict[str, Any]:
    """
    Resume únicamente los puntos actualmente visibles
    en el mapa.
    """

    return {
        "visible_patches": int(len(map_points)),
        "visible_groups": int(
            map_points["group_id"].nunique()
        ),
        "positive_patches": int(
            (map_points["etiqueta"] == 1).sum()
        ),
        "negative_patches": int(
            (map_points["etiqueta"] == 0).sum()
        ),
        "multipatch_groups": int(
            map_points.loc[
                map_points["es_multipatch"],
                "group_id",
            ].nunique()
        ),
    }