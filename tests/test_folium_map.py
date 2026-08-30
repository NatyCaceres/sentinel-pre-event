import folium
import pandas as pd
import pytest

from src.geo.folium_map import (
    MIXED_CLUSTER_COLOR,
    NEGATIVE_COLOR,
    POSITIVE_COLOR,
    Phase1MapError,
    build_phase1_folium_map,
    format_reference_date,
    get_cluster_color,
    get_marker_color,
)


def build_sample_points() -> pd.DataFrame:
    """
    Dataset mínimo compatible con el mapa.
    """

    return pd.DataFrame(
        {
            "patch_id": [
                "P_POS_001",
                "P_NEG_001",
            ],
            "group_id": [
                "EV_001",
                "NEG_001",
            ],
            "lat": [
                -37.5,
                -38.0,
            ],
            "lon": [
                -72.0,
                -72.5,
            ],
            "etiqueta": [
                1,
                0,
            ],
            "clase": [
                "Con detección posterior",
                "Sin detección cercana",
            ],
            "temporada": [
                "2024–2025",
                "2024–2025",
            ],
            "fecha_referencia": [
                pd.Timestamp("2025-01-15"),
                pd.Timestamp("2025-02-20"),
            ],
            "COMUNA": [
                "Comuna A",
                "Comuna B",
            ],
            "PROVINCIA": [
                "Provincia A",
                "Provincia B",
            ],
            "REGION": [
                "Región A",
                "Región B",
            ],
        }
    )


def test_cluster_color_for_all_classes() -> None:
    assert (
        get_cluster_color("Todas")
        == MIXED_CLUSTER_COLOR
    )

    assert (
        get_cluster_color(
            "Con detección posterior"
        )
        == POSITIVE_COLOR
    )

    assert (
        get_cluster_color(
            "Sin detección cercana"
        )
        == NEGATIVE_COLOR
    )


def test_marker_color_matches_label() -> None:
    assert get_marker_color(1) == POSITIVE_COLOR
    assert get_marker_color(0) == NEGATIVE_COLOR


def test_invalid_marker_label_is_rejected() -> None:
    with pytest.raises(Phase1MapError):
        get_marker_color(2)


def test_reference_date_is_formatted() -> None:
    result = format_reference_date(
        pd.Timestamp("2025-01-15")
    )

    assert result == "15-01-2025"


def test_empty_points_are_rejected() -> None:
    empty = build_sample_points().iloc[0:0]

    with pytest.raises(Phase1MapError):
        build_phase1_folium_map(
            empty,
            class_filter="Todas",
        )


def test_missing_required_column_is_rejected() -> None:
    dataframe = build_sample_points().drop(
        columns=["COMUNA"]
    )

    with pytest.raises(Phase1MapError):
        build_phase1_folium_map(
            dataframe,
            class_filter="Todas",
        )


def test_phase1_map_is_created() -> None:
    dataframe = build_sample_points()

    result = build_phase1_folium_map(
        dataframe,
        class_filter="Todas",
    )

    assert isinstance(
        result,
        folium.Map,
    )

    # Construir el mapa no debe alterar los datos recibidos.
    assert len(dataframe) == 2
    assert "geometry" not in dataframe.columns