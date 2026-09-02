import json
from pathlib import Path

import numpy as np
import pytest

from src.spectral.inference import (
    DEFAULT_INFERENCE_CONTRACT_PATH,
    SpectralInferenceError,
    load_inference_contract,
    load_spectral_model,
    predict_spectral_patch,
    preprocess_spectral_patch,
)
from src.spectral.patch import (
    SpectralPatch,
    SpectralPatchError,
    load_spectral_patch,
)
from src.spectral.storage import resolve_tif_path


CHANNELS = (
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "B12",
    "NDMI",
    "NBR",
)
SHAPE = (
    8,
    64,
    64,
)
REAL_TIFF_CASES = (
    (
        "model8__test__neg__P_2018_NEG_00015.tif",
        True,
    ),
    (
        "model8__test__pos__P_2019_POS_C0079_01.tif",
        True,
    ),
    (
        "model8__test__neg__P_2018_NEG_00014.tif",
        False,
    ),
)


def create_patch(
    data: np.ndarray,
    *,
    channels: tuple[str, ...] = CHANNELS,
) -> SpectralPatch:
    return SpectralPatch(
        data=data,
        channels=channels,
        path=Path("test_patch.tif"),
    )


@pytest.fixture(scope="module")
def loaded_model():
    contract = load_inference_contract()
    checkpoint_path = (
        DEFAULT_INFERENCE_CONTRACT_PATH.parent
        / contract["model"]["checkpoint_relative_path"]
    ).resolve()

    if not checkpoint_path.exists():
        pytest.skip(
            "El checkpoint local no está disponible."
        )

    return load_spectral_model()


def test_preprocessing_clips_each_channel() -> None:
    contract = load_inference_contract()
    preprocessing = contract["preprocessing"]
    normalization = preprocessing["normalization"]
    mean = np.asarray(
        normalization["mean"],
        dtype=np.float32,
    )
    std = np.asarray(
        normalization["std"],
        dtype=np.float32,
    )
    data = np.broadcast_to(
        mean[:, None, None],
        SHAPE,
    ).copy()

    for index, channel in enumerate(CHANNELS):
        lower, upper = preprocessing["clip_ranges"][channel]
        data[index, 0, 0] = lower - 10.0
        data[index, 0, 1] = upper + 10.0

    result = preprocess_spectral_patch(
        create_patch(data)
    )

    for index, channel in enumerate(CHANNELS):
        lower, upper = preprocessing["clip_ranges"][channel]

        assert result[index, 0, 0] == pytest.approx(
            (lower - mean[index]) / std[index]
        )
        assert result[index, 0, 1] == pytest.approx(
            (upper - mean[index]) / std[index]
        )

    assert result.shape == SHAPE
    assert result.dtype == np.float32
    assert result.flags.c_contiguous


def test_nan_is_zero_after_normalization() -> None:
    contract = load_inference_contract()
    mean = np.asarray(
        contract["preprocessing"]["normalization"]["mean"],
        dtype=np.float32,
    )
    data = np.broadcast_to(
        mean[:, None, None],
        SHAPE,
    ).copy()
    data[:, 10, 20] = np.nan

    result = preprocess_spectral_patch(
        create_patch(data)
    )

    assert np.count_nonzero(result) == 0
    assert np.isfinite(result).all()


def test_preprocessing_does_not_modify_original_data() -> None:
    data = np.zeros(
        SHAPE,
        dtype=np.float32,
    )
    data[0, 0, 0] = np.nan
    original = data.copy()

    preprocess_spectral_patch(
        create_patch(data)
    )

    np.testing.assert_equal(
        data,
        original,
    )


def test_infinity_is_rejected() -> None:
    data = np.zeros(
        SHAPE,
        dtype=np.float32,
    )
    data[0, 0, 0] = np.inf

    with pytest.raises(
        SpectralPatchError,
        match="infinitos",
    ):
        preprocess_spectral_patch(
            create_patch(data)
        )


def test_unexpected_channel_order_is_rejected() -> None:
    data = np.zeros(
        SHAPE,
        dtype=np.float32,
    )

    with pytest.raises(
        SpectralInferenceError,
        match="orden de canales",
    ):
        preprocess_spectral_patch(
            create_patch(
                data,
                channels=tuple(reversed(CHANNELS)),
            )
        )


def test_real_checkpoint_loads_strictly(
    loaded_model,
) -> None:
    assert loaded_model.checkpoint_path.name == (
        "best_resnet18_8ch_v2.pt"
    )
    assert loaded_model.device.type == "cpu"
    assert not loaded_model.model.training
    assert tuple(
        loaded_model.model.conv1.weight.shape
    ) == (
        64,
        8,
        7,
        7,
    )
    assert tuple(
        loaded_model.model.fc.weight.shape
    ) == (
        2,
        512,
    )


def test_prediction_returns_valid_probabilities(
    loaded_model,
) -> None:
    mean = np.asarray(
        loaded_model.contract[
            "preprocessing"
        ]["normalization"]["mean"],
        dtype=np.float32,
    )
    data = np.broadcast_to(
        mean[:, None, None],
        SHAPE,
    ).copy()
    data[:, 0, 0] = np.nan

    prediction = predict_spectral_patch(
        create_patch(data),
        loaded_model,
    )

    assert len(prediction.probabilities) == 2
    assert sum(prediction.probabilities) == pytest.approx(
        1.0
    )
    assert all(
        0.0 <= value <= 1.0
        for value in prediction.probabilities
    )
    assert prediction.class_1_score == (
        prediction.probabilities[1]
    )
    assert prediction.predicted_class == int(
        prediction.class_1_score
        >= prediction.threshold
    )
    assert prediction.threshold == pytest.approx(
        0.495
    )


def test_checkpoint_hash_is_checked_before_loading(
    tmp_path,
) -> None:
    contract = load_inference_contract()
    contract["model"]["checkpoint_relative_path"] = (
        "invalid_checkpoint.pt"
    )
    invalid_checkpoint_path = (
        tmp_path
        / "invalid_checkpoint.pt"
    )
    invalid_checkpoint_path.write_bytes(
        b"not a checkpoint"
    )
    contract_path = (
        tmp_path
        / "inference_contract.json"
    )
    contract_path.write_text(
        json.dumps(contract),
        encoding="utf-8",
    )

    with pytest.raises(
        SpectralInferenceError,
        match="SHA-256",
    ):
        load_spectral_model(
            contract_path
        )


@pytest.mark.parametrize(
    ("tif_filename", "expected_nan"),
    REAL_TIFF_CASES,
)
def test_real_tiff_end_to_end_inference(
    loaded_model,
    tif_filename: str,
    expected_nan: bool,
) -> None:
    try:
        tif_path = resolve_tif_path(
            tif_filename
        )
    except FileNotFoundError:
        pytest.skip(
            "Los GeoTIFF locales no están disponibles."
        )

    patch = load_spectral_patch(
        tif_path
    )

    assert bool(
        np.isnan(patch.data).any()
    ) is expected_nan

    prediction = predict_spectral_patch(
        patch,
        loaded_model,
    )

    assert len(prediction.probabilities) == 2
    assert sum(prediction.probabilities) == pytest.approx(
        1.0
    )
    assert all(
        np.isfinite(value)
        for value in prediction.probabilities
    )
    assert prediction.predicted_class in (0, 1)
