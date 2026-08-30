from __future__ import annotations

from typing import Any

import pandas as pd


def format_season_label(season_year: int) -> str:
    """
    Convierte el identificador interno de temporada
    en una etiqueta más clara para la interfaz.

    Ejemplo:
        2025 -> "2025–2026"
    """

    start_year = int(season_year)
    end_year = start_year + 1

    return f"{start_year}–{end_year}"


def format_class_label(label: int) -> str:
    """
    Convierte la clase numérica en una etiqueta
    comprensible para el usuario.
    """

    class_labels = {
        0: "Sin detección cercana",
        1: "Con detección posterior",
    }

    try:
        return class_labels[int(label)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Etiqueta de clase no válida: {label}"
        ) from exc


def build_map_points(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convierte metadata de Fase 1 en una tabla preparada
    para ser consumida por el futuro mapa.

    Esta función no modifica el DataFrame original.
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

    # Cantidad de parches asociados a cada grupo/evento.
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

    # Etiqueta visual de temporada.
    points["temporada"] = (
        pd.to_numeric(
            points["season_year"],
            errors="raise",
        )
        .astype(int)
        .map(format_season_label)
    )

    # Etiqueta visual de clase.
    points["clase"] = (
        pd.to_numeric(
            points["etiqueta"],
            errors="raise",
        )
        .astype(int)
        .map(format_class_label)
    )

    # Convertimos la fecha a un tipo temporal real.
    points["fecha_referencia"] = (
        pd.to_datetime(
            points["fecha_referencia"],
            errors="raise",
        )
        .dt.date
    )

    # Nombres más apropiados para la capa geográfica.
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
    Calcula las métricas correspondientes únicamente
    a los puntos actualmente visibles en el mapa.
    """

    return {
        "visible_patches": int(
            len(map_points)
        ),
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