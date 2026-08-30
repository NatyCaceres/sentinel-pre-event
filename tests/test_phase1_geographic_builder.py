import pandas as pd
import pytest

from src.geo.build_phase1_geographic import (
    GeographicBuildError,
    validate_derived_dataset,
)


def test_derived_dataset_preserves_identity() -> None:
    original = pd.DataFrame(
        {
            "patch_id": [
                "P_001",
                "P_002",
            ]
        }
    )

    enriched = pd.DataFrame(
        {
            "patch_id": [
                "P_001",
                "P_002",
            ],
            "COMUNA": [
                "Comuna A",
                "Comuna B",
            ],
        }
    )

    validate_derived_dataset(
        original,
        enriched,
    )


def test_derived_dataset_rejects_missing_rows() -> None:
    original = pd.DataFrame(
        {
            "patch_id": [
                "P_001",
                "P_002",
            ]
        }
    )

    enriched = pd.DataFrame(
        {
            "patch_id": [
                "P_001",
            ]
        }
    )

    with pytest.raises(GeographicBuildError):
        validate_derived_dataset(
            original,
            enriched,
        )


def test_derived_dataset_rejects_duplicates() -> None:
    original = pd.DataFrame(
        {
            "patch_id": [
                "P_001",
                "P_002",
            ]
        }
    )

    enriched = pd.DataFrame(
        {
            "patch_id": [
                "P_001",
                "P_001",
            ]
        }
    )

    with pytest.raises(GeographicBuildError):
        validate_derived_dataset(
            original,
            enriched,
        )


def test_derived_dataset_rejects_changed_ids() -> None:
    original = pd.DataFrame(
        {
            "patch_id": [
                "P_001",
                "P_002",
            ]
        }
    )

    enriched = pd.DataFrame(
        {
            "patch_id": [
                "P_001",
                "P_999",
            ]
        }
    )

    with pytest.raises(GeographicBuildError):
        validate_derived_dataset(
            original,
            enriched,
        )