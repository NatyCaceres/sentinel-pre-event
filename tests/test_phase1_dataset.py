import pandas as pd
import pytest

from src.data.phase1 import (
    PatchNotFoundError,
    Phase1DataError,
    Phase1Dataset,
)


def build_dataset() -> Phase1Dataset:
    """
    Construye un Phase1Dataset sintético para probar
    las consultas sin depender del CSV real.
    """

    dataframe = pd.DataFrame(
        {
            "patch_id": [
                "P_001",
                "P_002",
                "P_003",
                "P_004",
                "P_005",
            ],
            "group_id": [
                "EV_001",
                "EV_001",
                "EV_002",
                "EV_003",
                "EV_004",
            ],
            "etiqueta": [
                1,
                1,
                0,
                1,
                0,
            ],
            "lat_centro": [
                -37.1,
                -37.2,
                -38.0,
                -39.0,
                -40.0,
            ],
            "lon_centro": [
                -72.1,
                -72.2,
                -73.0,
                -72.5,
                -71.5,
            ],
            "season_year": [
                2024,
                2024,
                2024,
                2025,
                2025,
            ],
            "banda_lat": [
                "centro",
                "centro",
                "sur",
                "norte",
                "sur",
            ],
            "fecha_referencia": [
                "2025-01-10",
                "2025-01-10",
                "2025-02-15",
                "2026-01-20",
                "2026-03-05",
            ],
            "fecha_objetivo_pre_evento": [
                "2024-10-10",
                "2024-10-10",
                "2024-11-15",
                "2025-10-20",
                "2025-12-05",
            ],
            "gee_inicio": [
                "2024-10-01",
                "2024-10-01",
                "2024-11-01",
                "2025-10-01",
                "2025-12-01",
            ],
            "gee_fin": [
                "2024-11-01",
                "2024-11-01",
                "2024-12-01",
                "2025-11-01",
                "2026-01-01",
            ],
            "spatial_block": [
                "B01",
                "B01",
                "B02",
                "B03",
                "B04",
            ],
            "split": [
                "train",
                "train",
                "test",
                "train",
                "test",
            ],
            "event_weight": [
                0.5,
                0.5,
                1.0,
                1.0,
                1.0,
            ],
        }
    )

    manifest = {
        "dataset_id": "phase1_test_dataset",
    }

    return Phase1Dataset(
        dataframe=dataframe,
        dataset_path=pd.NA,
        manifest=manifest,
    )


def test_get_patch_returns_expected_patch() -> None:
    dataset = build_dataset()

    patch = dataset.get_patch("P_003")

    assert patch["patch_id"] == "P_003"
    assert patch["group_id"] == "EV_002"
    assert patch["etiqueta"] == 0


def test_get_patch_raises_when_not_found() -> None:
    dataset = build_dataset()

    with pytest.raises(PatchNotFoundError):
        dataset.get_patch("P_INEXISTENTE")


def test_get_group_patches_returns_all_patches() -> None:
    dataset = build_dataset()

    result = dataset.get_group_patches("EV_001")

    assert len(result) == 2
    assert set(result["patch_id"]) == {
        "P_001",
        "P_002",
    }


def test_available_filters_are_derived_from_data() -> None:
    dataset = build_dataset()

    filters = dataset.get_available_filters()

    assert filters["seasons"] == [2024, 2025]
    assert filters["labels"] == [0, 1]

    assert set(filters["latitude_bands"]) == {
        "centro",
        "norte",
        "sur",
    }

    assert filters["splits"] == [
        "test",
        "train",
    ]

    assert filters["reference_date_min"] == "2025-01-10"
    assert filters["reference_date_max"] == "2026-03-05"


def test_combined_filter_works() -> None:
    dataset = build_dataset()

    result = dataset.filter(
        season_year=2025,
        label=1,
        split="train",
    )

    assert len(result) == 1
    assert result.iloc[0]["patch_id"] == "P_004"


def test_reference_date_filter_works() -> None:
    dataset = build_dataset()

    result = dataset.filter(
        reference_date_start="2026-01-01",
        reference_date_end="2026-12-31",
    )

    assert set(result["patch_id"]) == {
        "P_004",
        "P_005",
    }


def test_multipatch_filter_works() -> None:
    dataset = build_dataset()

    result = dataset.filter(
        multipatch_only=True,
    )

    assert len(result) == 2
    assert result["group_id"].nunique() == 1
    assert result.iloc[0]["group_id"] == "EV_001"


def test_invalid_label_is_rejected() -> None:
    dataset = build_dataset()

    with pytest.raises(Phase1DataError):
        dataset.filter(label=9)