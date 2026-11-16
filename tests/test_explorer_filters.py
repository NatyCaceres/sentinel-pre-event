import pandas as pd
import pytest

from src.data.explorer_filters import (
    ExplorerFilterError,
    apply_explorer_filters,
    get_reference_date_bounds,
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


@pytest.mark.parametrize("label", [0, 1])
def test_filter_by_label(label) -> None:
    points = build_sample_points()

    result = apply_explorer_filters(
        points,
        label=label,
    )

    assert len(result) == 2

    assert (
        result["etiqueta"] == label
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


def test_date_bounds_follow_selected_season() -> None:
    points = build_sample_points()
    assert get_reference_date_bounds(points, 2024) == (
        pd.Timestamp("2025-01-10").date(),
        pd.Timestamp("2025-02-10").date(),
    )
    assert get_reference_date_bounds(points, 2025) == (
        pd.Timestamp("2026-01-15").date(),
        pd.Timestamp("2026-02-20").date(),
    )
    assert get_reference_date_bounds(points) == (
        pd.Timestamp("2025-01-10").date(),
        pd.Timestamp("2026-02-20").date(),
    )


def test_date_bounds_reject_season_without_dates() -> None:
    with pytest.raises(ExplorerFilterError):
        get_reference_date_bounds(build_sample_points(), 2030)


def test_unmapped_sample_counts_only_when_it_matches_query() -> None:
    points = build_sample_points()
    points["map_eligible"] = [True, True, True, False]
    points.loc[3, "REGION"] = None

    result = apply_explorer_filters(points, season_year=2025, label=0)
    assert result["patch_id"].tolist() == ["P_004"]
    assert result.loc[result["map_eligible"]].empty

    assert apply_explorer_filters(
        points, season_year=2024, label=0
    )["patch_id"].tolist() == ["P_002"]
    assert apply_explorer_filters(
        points, season_year=2025, label=0, region="La Araucanía"
    ).empty
    assert apply_explorer_filters(
        points, season_year=2025, reference_end="2026-01-31"
    )["patch_id"].tolist() == ["P_003"]
