import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

import src.geo.administrative_boundaries as boundaries
from src.geo.administrative_boundaries import (
    AdministrativeBoundaryError,
    build_geographic_points,
    enrich_with_administrative_boundaries,
)


def build_sample_communes() -> gpd.GeoDataFrame:
    """
    Construye polígonos comunales sintéticos para
    probar la lógica espacial sin depender de la DPA real.
    """

    commune_a = Polygon(
        [
            (-73.0, -38.0),
            (-71.0, -38.0),
            (-71.0, -36.0),
            (-73.0, -36.0),
        ]
    )

    commune_b = Polygon(
        [
            (-73.0, -40.0),
            (-71.0, -40.0),
            (-71.0, -38.0),
            (-73.0, -38.0),
        ]
    )

    return gpd.GeoDataFrame(
        {
            "CUT_REG": ["08", "09"],
            "CUT_PROV": ["083", "091"],
            "CUT_COM": ["08301", "09101"],
            "REGION": [
                "Biobío",
                "La Araucanía",
            ],
            "PROVINCIA": [
                "Biobío",
                "Cautín",
            ],
            "COMUNA": [
                "Comuna A",
                "Comuna B",
            ],
            "geometry": [
                commune_a,
                commune_b,
            ],
        },
        crs="EPSG:4326",
    )


def build_sample_points() -> pd.DataFrame:
    """
    Construye tres puntos:
    - uno dentro de Comuna A;
    - uno dentro de Comuna B;
    - uno fuera de ambas.
    """

    return pd.DataFrame(
        {
            "patch_id": [
                "P_001",
                "P_002",
                "P_003",
            ],
            "lat_centro": [
                -37.0,
                -39.0,
                -33.0,
            ],
            "lon_centro": [
                -72.0,
                -72.0,
                -72.0,
            ],
        }
    )


def test_build_geographic_points() -> None:
    dataframe = build_sample_points()

    points = build_geographic_points(dataframe)

    assert isinstance(
        points,
        gpd.GeoDataFrame,
    )

    assert points.crs.to_string() == "EPSG:4326"

    assert (
        points.geometry.geom_type
        == "Point"
    ).all()

    # El DataFrame original no debe ser modificado.
    assert "geometry" not in dataframe.columns


def test_enrichment_assigns_communes(
    monkeypatch,
) -> None:
    dataframe = build_sample_points()
    communes = build_sample_communes()

    monkeypatch.setattr(
        boundaries,
        "load_communes",
        lambda path: communes,
    )

    result = enrich_with_administrative_boundaries(
        dataframe
    )

    assert len(result) == 3

    patch_1 = result[
        result["patch_id"] == "P_001"
    ].iloc[0]

    patch_2 = result[
        result["patch_id"] == "P_002"
    ].iloc[0]

    assert patch_1["COMUNA"] == "Comuna A"
    assert patch_1["REGION"] == "Biobío"

    assert patch_2["COMUNA"] == "Comuna B"
    assert patch_2["REGION"] == "La Araucanía"

    assert (
        patch_1["administrative_status"]
        == "assigned"
    )


def test_unresolved_point_is_not_forced(
    monkeypatch,
) -> None:
    dataframe = build_sample_points()
    communes = build_sample_communes()

    monkeypatch.setattr(
        boundaries,
        "load_communes",
        lambda path: communes,
    )

    result = enrich_with_administrative_boundaries(
        dataframe
    )

    unresolved = result[
        result["patch_id"] == "P_003"
    ].iloc[0]

    assert (
        unresolved["administrative_status"]
        == "unresolved"
    )

    assert pd.isna(
        unresolved["COMUNA"]
    )

    assert pd.isna(
        unresolved["REGION"]
    )


def test_missing_coordinates_are_rejected() -> None:
    dataframe = build_sample_points()

    dataframe = dataframe.drop(
        columns=["lon_centro"]
    )

    with pytest.raises(
        AdministrativeBoundaryError
    ):
        build_geographic_points(dataframe)