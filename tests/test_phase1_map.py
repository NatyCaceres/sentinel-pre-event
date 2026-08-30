import pandas as pd
import pytest

from src.geo.phase1_map import (
    build_map_points,
    format_class_label,
    format_season_label,
    get_map_summary,
)


def build_sample_dataframe() -> pd.DataFrame:
    """
    Dataset geográfico sintético para probar la
    preparación de puntos del mapa.
    """

    return pd.DataFrame(
        {
            "patch_id": [
                "P_001",
                "P_002",
                "P_003",
            ],
            "group_id": [
                "EV_001",
                "EV_001",
                "EV_002",
            ],
            "etiqueta": [
                1,
                1,
                0,
            ],
            "lat_centro": [
                -37.1,
                -37.2,
                -39.0,
            ],
            "lon_centro": [
                -72.1,
                -72.2,
                -73.0,
            ],
            "season_year": [
                2025,
                2025,
                2024,
            ],
            "banda_lat": [
                "centro",
                "centro",
                "sur",
            ],
            "fecha_referencia": [
                "2026-01-20",
                "2026-01-20",
                "2025-02-10",
            ],
            "split": [
                "train",
                "train",
                "test",
            ],
        }
    )


def test_format_season_label() -> None:
    assert format_season_label(2025) == "2025–2026"


def test_format_class_label() -> None:
    assert (
        format_class_label(1)
        == "Con detección posterior"
    )

    assert (
        format_class_label(0)
        == "Sin detección cercana"
    )


def test_build_map_points_detects_multipatch() -> None:
    df = build_sample_dataframe()

    points = build_map_points(df)

    event_1 = points[
        points["group_id"] == "EV_001"
    ]

    event_2 = points[
        points["group_id"] == "EV_002"
    ]

    assert set(event_1["n_patches_grupo"]) == {2}
    assert event_1["es_multipatch"].all()

    assert set(event_2["n_patches_grupo"]) == {1}
    assert not event_2["es_multipatch"].any()


def test_build_map_points_does_not_modify_input() -> None:
    df = build_sample_dataframe()

    build_map_points(df)

    assert "lat_centro" in df.columns
    assert "lon_centro" in df.columns

    assert "lat" not in df.columns
    assert "lon" not in df.columns


def test_map_summary() -> None:
    df = build_sample_dataframe()

    points = build_map_points(df)

    summary = get_map_summary(points)

    assert summary == {
        "visible_patches": 3,
        "visible_groups": 2,
        "positive_patches": 2,
        "negative_patches": 1,
        "multipatch_groups": 1,
    }


def test_missing_columns_are_rejected() -> None:
    df = build_sample_dataframe()

    df = df.drop(
        columns=["lon_centro"]
    )

    with pytest.raises(ValueError):
        build_map_points(df)