import pandas as pd

from src.data.metadata_validator import (
    load_phase1_schema,
    validate_phase1_metadata,
)


def build_valid_metadata() -> pd.DataFrame:
    """
    Construye un dataset mínimo sintético que cumple
    el contrato de Fase 1.
    """

    return pd.DataFrame(
        {
            "patch_id": [
                "P_TEST_001",
                "P_TEST_002",
                "P_TEST_003",
                "P_TEST_004",
            ],
            "group_id": [
                "G_TEST_001",
                "G_TEST_002",
                "G_TEST_003",
                "G_TEST_004",
            ],
            "etiqueta": [
                1,
                0,
                1,
                0,
            ],
            "lat_centro": [
                -37.50,
                -37.60,
                -38.10,
                -38.20,
            ],
            "lon_centro": [
                -72.50,
                -72.60,
                -73.10,
                -73.20,
            ],
            "season_year": [
                2025,
                2025,
                2025,
                2025,
            ],
            "banda_lat": [
                "centro",
                "centro",
                "sur",
                "sur",
            ],
            "fecha_referencia": [
                "2025-03-01",
                "2025-03-01",
                "2025-03-15",
                "2025-03-15",
            ],
            "fecha_objetivo_pre_evento": [
                "2024-12-01",
                "2024-12-01",
                "2024-12-15",
                "2024-12-15",
            ],
            "gee_inicio": [
                "2024-12-01",
                "2024-12-01",
                "2024-12-01",
                "2024-12-01",
            ],
            "gee_fin": [
                "2025-01-01",
                "2025-01-01",
                "2025-01-01",
                "2025-01-01",
            ],
            "spatial_block": [
                "BLOCK_001",
                "BLOCK_002",
                "BLOCK_003",
                "BLOCK_004",
            ],
            "split": [
                "train",
                "train",
                "test",
                "test",
            ],
            "event_weight": [
                1.0,
                1.0,
                1.0,
                1.0,
            ],
        }
    )


def test_valid_metadata_passes() -> None:
    schema = load_phase1_schema()
    df = build_valid_metadata()

    report = validate_phase1_metadata(
        df,
        schema,
    )

    assert report.is_valid
    assert report.errors == []


def test_duplicate_patch_id_is_detected() -> None:
    schema = load_phase1_schema()
    df = build_valid_metadata()

    df.loc[1, "patch_id"] = df.loc[0, "patch_id"]

    report = validate_phase1_metadata(
        df,
        schema,
    )

    assert not report.is_valid

    assert any(
        "patch_id" in error
        and "único" in error
        for error in report.errors
    )


def test_group_leakage_is_detected() -> None:
    schema = load_phase1_schema()
    df = build_valid_metadata()

    # El grupo de una muestra de test pasa a ser
    # el mismo de una muestra de train.
    df.loc[2, "group_id"] = df.loc[0, "group_id"]

    report = validate_phase1_metadata(
        df,
        schema,
    )

    assert not report.is_valid

    assert any(
        "Fuga de información" in error
        and "group_id" in error
        for error in report.errors
    )


def test_spatial_block_leakage_is_detected() -> None:
    schema = load_phase1_schema()
    df = build_valid_metadata()

    df.loc[2, "spatial_block"] = (
        df.loc[0, "spatial_block"]
    )

    report = validate_phase1_metadata(
        df,
        schema,
    )

    assert not report.is_valid

    assert any(
        "Fuga de información" in error
        and "spatial_block" in error
        for error in report.errors
    )


def test_invalid_coordinates_are_detected() -> None:
    schema = load_phase1_schema()
    df = build_valid_metadata()

    df.loc[0, "lat_centro"] = -60.0

    report = validate_phase1_metadata(
        df,
        schema,
    )

    assert not report.is_valid

    assert any(
        "latitud" in error
        for error in report.errors
    )