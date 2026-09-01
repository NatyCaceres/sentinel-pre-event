import pandas as pd
import pytest

from src.spectral.storage import (
    SpectralStorageError,
    get_patch_directory,
    resolve_patch_tif,
    resolve_tif_path,
)


def test_default_patch_directory_exists_as_path() -> None:
    result = get_patch_directory()

    assert result.name == "sentinel_8ch"


def test_environment_directory_is_used(
    tmp_path,
    monkeypatch,
) -> None:

    monkeypatch.setenv(
        "SENTINEL_PATCH_DIR",
        str(tmp_path),
    )

    result = get_patch_directory()

    assert result == tmp_path.resolve()


def test_resolve_existing_tif(
    tmp_path,
    monkeypatch,
) -> None:

    monkeypatch.setenv(
        "SENTINEL_PATCH_DIR",
        str(tmp_path),
    )

    tif = tmp_path / "patch.tif"
    tif.write_bytes(b"test")

    result = resolve_tif_path(
        "patch.tif"
    )

    assert result == tif


def test_missing_tif_is_rejected(
    tmp_path,
    monkeypatch,
) -> None:

    monkeypatch.setenv(
        "SENTINEL_PATCH_DIR",
        str(tmp_path),
    )

    with pytest.raises(
        FileNotFoundError
    ):
        resolve_tif_path(
            "missing.tif"
        )


def test_path_instead_of_filename_is_rejected() -> None:

    with pytest.raises(
        SpectralStorageError
    ):
        resolve_tif_path(
            "folder/patch.tif",
            require_exists=False,
        )


def test_invalid_extension_is_rejected() -> None:

    with pytest.raises(
        SpectralStorageError
    ):
        resolve_tif_path(
            "patch.csv",
            require_exists=False,
        )


def test_resolve_patch_series(
    tmp_path,
    monkeypatch,
) -> None:

    monkeypatch.setenv(
        "SENTINEL_PATCH_DIR",
        str(tmp_path),
    )

    tif = tmp_path / "P_001.tif"
    tif.write_bytes(b"test")

    patch = pd.Series(
        {
            "patch_id": "P_001",
            "tif_filename": "P_001.tif",
        }
    )

    result = resolve_patch_tif(
        patch
    )

    assert result == tif