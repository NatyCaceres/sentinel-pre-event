import pandas as pd
import pytest

from src.data.product_catalog import (
    ProductCatalogError,
    build_product_catalog,
)


def build_phase1() -> pd.DataFrame:
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
                "NEG_001",
            ],
            "etiqueta": [
                1,
                1,
                0,
            ],
            "lat_centro": [
                -37.0,
                -37.1,
                -38.0,
            ],
            "lon_centro": [
                -72.0,
                -72.1,
                -73.0,
            ],
            "season_year": [
                2025,
                2025,
                2025,
            ],
            "fecha_referencia": [
                "2026-01-10",
                "2026-01-10",
                "2026-02-10",
            ],
            "fecha_objetivo_pre_evento": [
                "2025-10-10",
                "2025-10-10",
                "2025-11-10",
            ],
            "split": [
                "train",
                "train",
                "test",
            ],
            "event_weight": [
                0.5,
                0.5,
                1.0,
            ],
            "COMUNA": [
                "Comuna A",
                "Comuna A",
                None,
            ],
            "PROVINCIA": [
                "Provincia A",
                "Provincia A",
                None,
            ],
            "REGION": [
                "Región A",
                "Región A",
                None,
            ],
            "administrative_status": [
                "assigned",
                "assigned",
                "unresolved",
            ],
        }
    )


def build_phase2() -> pd.DataFrame:
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
                "NEG_001",
            ],
            "etiqueta": [
                1,
                1,
                0,
            ],
            "lat_centro": [
                -37.0,
                -37.1,
                -38.0,
            ],
            "lon_centro": [
                -72.0,
                -72.1,
                -73.0,
            ],
            "season_year": [
                2025,
                2025,
                2025,
            ],
            "banda_lat": [
                "centro",
                "centro",
                "sur",
            ],
            "fecha_referencia": [
                "2026-01-10",
                "2026-01-10",
                "2026-02-10",
            ],
            "fecha_objetivo_pre_evento": [
                "2025-10-10",
                "2025-10-10",
                "2025-11-10",
            ],
            "gee_inicio": [
                "2025-10-01",
                "2025-10-01",
                "2025-11-01",
            ],
            "gee_fin": [
                "2025-10-31",
                "2025-10-31",
                "2025-11-30",
            ],
            "spatial_block": [
                "A",
                "A",
                "B",
            ],
            "split": [
                "train",
                "train",
                "test",
            ],
            "event_weight": [
                0.5,
                0.5,
                1.0,
            ],
            "image_id": [
                "IMG_1",
                "IMG_2",
                "IMG_3",
            ],
            "fecha_imagen_seleccionada": [
                "2025-10-12",
                "2025-10-12",
                "2025-11-12",
            ],
            "dias_diferencia_objetivo": [
                2,
                2,
                2,
            ],
            "pct_pixeles_validos_patch": [
                100.0,
                99.0,
                98.0,
            ],
            "cloudy_pixel_percentage": [
                1.0,
                2.0,
                3.0,
            ],
            "tipo_seleccion": [
                "single",
                "single",
                "single",
            ],
            "n_assets_candidatos": [
                3,
                3,
                2,
            ],
            "n_imagenes_candidatas": [
                3,
                3,
                2,
            ],
            "n_imagenes_validas": [
                2,
                2,
                1,
            ],
            "archivo_model8_esperado": [
                "P_001.tif",
                "P_002.tif",
                "P_003.tif",
            ],
            "ruta_model_8ch": [
                "/drive/P_001.tif",
                "/drive/P_002.tif",
                "/drive/P_003.tif",
            ],
            "existe_tif": [
                True,
                True,
                True,
            ],
            "subsplit": [
                "train",
                "validation",
                "test",
            ],
            "event_weight_model": [
                0.5,
                0.5,
                1.0,
            ],
        }
    )


def test_catalog_preserves_phase2_rows() -> None:
    catalog = build_product_catalog(
        build_phase1(),
        build_phase2(),
    )

    assert len(catalog) == 3
    assert catalog["patch_id"].nunique() == 3


def test_catalog_adds_geography() -> None:
    catalog = build_product_catalog(
        build_phase1(),
        build_phase2(),
    )

    first = catalog.iloc[0]

    assert first["COMUNA"] == "Comuna A"
    assert first["REGION"] == "Región A"


def test_catalog_marks_map_eligibility() -> None:
    catalog = build_product_catalog(
        build_phase1(),
        build_phase2(),
    )

    assert int(
        catalog["map_eligible"].sum()
    ) == 2

    unresolved = catalog[
        catalog["patch_id"] == "P_003"
    ].iloc[0]

    assert not bool(
        unresolved["map_eligible"]
    )


def test_catalog_recalculates_available_multipatch() -> None:
    catalog = build_product_catalog(
        build_phase1(),
        build_phase2(),
    )

    event = catalog[
        catalog["group_id"] == "EV_001"
    ]

    assert (
        event[
            "n_patches_group_available"
        ] == 2
    ).all()

    assert (
        event[
            "es_multipatch_available"
        ]
    ).all()


def test_incompatible_phase2_is_rejected() -> None:
    phase1 = build_phase1()
    phase2 = build_phase2()

    phase2.loc[
        phase2["patch_id"] == "P_001",
        "etiqueta",
    ] = 0

    with pytest.raises(
        ProductCatalogError
    ):
        build_product_catalog(
            phase1,
            phase2,
        )