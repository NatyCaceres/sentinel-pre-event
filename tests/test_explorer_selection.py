import pandas as pd
import pytest

from src.data.explorer_selection import (
    GroupNotFoundError,
    PatchNotFoundError,
    get_available_group_ids,
    get_available_patch_ids,
    get_group,
    get_group_patch_ids,
    get_patch,
)


def build_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patch_id": [
                "P_003",
                "P_001",
                "P_002",
            ],
            "group_id": [
                "NEG_001",
                "EV_001",
                "EV_001",
            ],
            "etiqueta": [
                0,
                1,
                1,
            ],
        }
    )


def test_get_patch() -> None:
    catalog = build_catalog()

    patch = get_patch(
        catalog,
        "P_001",
    )

    assert patch["group_id"] == "EV_001"
    assert patch["etiqueta"] == 1


def test_missing_patch_is_rejected() -> None:
    catalog = build_catalog()

    with pytest.raises(
        PatchNotFoundError
    ):
        get_patch(
            catalog,
            "P_999",
        )


def test_get_group() -> None:
    catalog = build_catalog()

    group = get_group(
        catalog,
        "EV_001",
    )

    assert len(group) == 2

    assert set(
        group["patch_id"]
    ) == {
        "P_001",
        "P_002",
    }


def test_missing_group_is_rejected() -> None:
    catalog = build_catalog()

    with pytest.raises(
        GroupNotFoundError
    ):
        get_group(
            catalog,
            "EV_999",
        )


def test_available_patch_ids_are_sorted() -> None:
    catalog = build_catalog()

    result = get_available_patch_ids(
        catalog
    )

    assert result == [
        "P_001",
        "P_002",
        "P_003",
    ]


def test_available_group_ids_are_sorted() -> None:
    catalog = build_catalog()

    result = get_available_group_ids(
        catalog
    )

    assert result == [
        "EV_001",
        "NEG_001",
    ]


def test_group_patch_ids() -> None:
    catalog = build_catalog()

    result = get_group_patch_ids(
        catalog,
        "EV_001",
    )

    assert result == [
        "P_001",
        "P_002",
    ]