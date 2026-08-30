import pandas as pd
import pytest

from src.data.explorer_filters import (
    ExplorerFilterError,
    apply_explorer_filters,
)


def build_sample_points() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patch_id": [
                "P_001",
                "P_002",
                "P_003",
                "P_004",
            ],
            "season_year": [
                2024,
                2024,
                2025,
                2025,
            ],
            "REGION": [
                "Biobío",
                "Biobío",
                "La Araucanía",
                "La Araucanía",
            ],
            "etiqueta": [
                1,
                0,
                1,
                0,
            ],
            "fecha_referencia": pd.to_datetime(
                [
                    "2025-01-10",
                    "2025-02-10",
                    "2026-01-15",
                    "2026-02-20",
                ]
            ),
            "es_multipatch": [
                True,
                False,
                True,
                False,
            ],
        }
    )


def test_filter_by_season() -> None:
    points = build_sample_points()

    result = apply_explorer_filters(
        points,
        season_year=2025,
    )

    assert len(result) == 2

    assert set(
        result["patch_id"]
    ) == {
        "P_003",
        "P_004",
    }


def test_filter_by_region() -> None:
    points = build_sample_points()

    result = apply_explorer_filters(
        points,
        region="Biobío",
    )

    assert len(result) == 2

    assert (
        result["REGION"] == "Biobío"
    ).all()


def test_filter_by_label() -> None:
    points = build_sample_points()

    result = apply_explorer_filters(
        points,
        label=1,
    )

    assert len(result) == 2

    assert (
        result["etiqueta"] == 1
    ).all()


def test_filter_by_reference_dates() -> None:
    points = build_sample_points()

    result = apply_explorer_filters(
        points,
        reference_start="2025-02-01",
        reference_end="2026-01-31",
    )

    assert set(
        result["patch_id"]
    ) == {
        "P_002",
        "P_003",
    }


def test_filter_multipatch_only() -> None:
    points = build_sample_points()

    result = apply_explorer_filters(
        points,
        multipatch_only=True,
    )

    assert set(
        result["patch_id"]
    ) == {
        "P_001",
        "P_003",
    }


def test_combined_filters() -> None:
    points = build_sample_points()

    result = apply_explorer_filters(
        points,
        season_year=2025,
        region="La Araucanía",
        label=1,
        multipatch_only=True,
    )

    assert len(result) == 1

    assert (
        result.iloc[0]["patch_id"]
        == "P_003"
    )


def test_invalid_label_is_rejected() -> None:
    points = build_sample_points()

    with pytest.raises(ExplorerFilterError):
        apply_explorer_filters(
            points,
            label=5,
        )


def test_original_dataframe_is_not_modified() -> None:
    points = build_sample_points()

    original = points.copy(
        deep=True
    )

    apply_explorer_filters(
        points,
        season_year=2025,
        label=1,
    )

    pd.testing.assert_frame_equal(
        points,
        original,
    )