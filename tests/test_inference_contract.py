from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "resnet18_8ch_v2"
)
RUNTIME_DIRECTORY = MODEL_DIRECTORY / "runtime"
INFERENCE_CONTRACT_PATH = (
    RUNTIME_DIRECTORY
    / "inference_contract.json"
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def test_input_matches_spectral_contract() -> None:
    inference = load_json(
        INFERENCE_CONTRACT_PATH
    )
    spectral = load_json(
        PROJECT_ROOT
        / "config"
        / "spectral_contract.json"
    )

    assert inference["input"]["product"] == spectral["product"]
    assert inference["input"]["shape"] == spectral["shape"]
    assert inference["input"]["dtype"] == spectral["dtype"]
    assert inference["input"]["channels"] == spectral["channels"]
    assert (
        inference["preprocessing"]["clip_ranges"]
        == spectral["band_ranges"]
    )


def test_model_matches_experiment_config() -> None:
    inference = load_json(
        INFERENCE_CONTRACT_PATH
    )
    experiment = load_json(
        RUNTIME_DIRECTORY
        / "experiment_config.json"
    )
    model = inference["model"]

    assert model["architecture"] == experiment["model_name"]
    assert model["in_chans"] == experiment["in_chans"]
    assert model["num_classes"] == 2
    assert model["dropout_rate"] == experiment["dropout_rate"]
    assert inference["input"]["channels"] == experiment["channels"]
    assert inference["input"]["shape"] == experiment["input_size"]


def test_normalization_is_complete_and_train_only() -> None:
    inference = load_json(
        INFERENCE_CONTRACT_PATH
    )
    experiment = load_json(
        RUNTIME_DIRECTORY
        / "experiment_config.json"
    )
    preprocessing = inference["preprocessing"]
    normalization = preprocessing["normalization"]
    channel_count = len(inference["input"]["channels"])

    assert normalization["method"] == "zscore"
    assert normalization["statistics_source"].startswith(
        "train_only"
    )
    assert (
        normalization["source_sha256"]
        == experiment["normalization_sha256"]
    )
    assert len(normalization["mean"]) == channel_count
    assert len(normalization["std"]) == channel_count
    assert all(
        math.isfinite(value)
        for value in normalization["mean"]
    )
    assert all(
        math.isfinite(value) and value > 0
        for value in normalization["std"]
    )
    assert (
        preprocessing["nonfinite_policy"]["nan"]
        == "impute_train_channel_mean"
    )
    assert (
        preprocessing["nonfinite_policy"]
        ["imputed_value_after_zscore"]
        == 0.0
    )


def test_output_matches_validation_threshold() -> None:
    inference = load_json(
        INFERENCE_CONTRACT_PATH
    )
    threshold = load_json(
        RUNTIME_DIRECTORY
        / "validation_threshold.json"
    )
    output = inference["output"]

    assert output["activation"] == "softmax"
    assert [
        item["index"]
        for item in output["class_order"]
    ] == [0, 1]
    assert (
        output["decision_threshold"]
        == threshold["best_threshold"]
    )
    assert output["threshold_selected_on"] == threshold["selected_on"]
    assert output["threshold_criterion"] == threshold["criterion"]
    assert "no predice incendios" in output["interpretation_notice"]
    assert "riesgo absoluto" in output["interpretation_notice"]


def test_checkpoint_hash_matches_when_available() -> None:
    inference = load_json(
        INFERENCE_CONTRACT_PATH
    )
    model = inference["model"]
    checkpoint_path = (
        RUNTIME_DIRECTORY
        / model["checkpoint_relative_path"]
    ).resolve()

    if not checkpoint_path.exists():
        return

    assert checkpoint_path.parent == MODEL_DIRECTORY.resolve()
    assert sha256(checkpoint_path) == model["checkpoint_sha256"]
