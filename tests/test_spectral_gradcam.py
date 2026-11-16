from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import timm
import torch
from torch.nn import functional as F

from src.spectral.gradcam import SpectralGradCAMError, calculate_gradcam
from src.spectral.inference import (
    DEFAULT_INFERENCE_CONTRACT_PATH,
    LoadedSpectralModel,
    load_inference_contract,
    load_spectral_model,
    predict_spectral_patch,
    preprocess_spectral_patch,
)
from src.spectral.patch import SpectralPatch, load_spectral_patch
from src.spectral.storage import resolve_tif_path


@pytest.fixture
def loaded_model():
    # Sin checkpoint externo ni descargas: las pruebas básicas siempre corren.
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = timm.create_model("resnet18", pretrained=False, in_chans=8, num_classes=2)
    model.eval()
    return LoadedSpectralModel(
        model=model,
        contract=load_inference_contract(),
        contract_path=DEFAULT_INFERENCE_CONTRACT_PATH,
        checkpoint_path=Path("unused.pt"),
        device=torch.device("cpu"),
    )


@pytest.fixture
def patch():
    data = np.random.default_rng(42).uniform(0, 1, (8, 64, 64)).astype(np.float32)
    data[:, 3, 5] = np.nan
    return SpectralPatch(
        data=data,
        channels=tuple(load_inference_contract()["input"]["channels"]),
        path=Path("synthetic.tif"),
    )


def notebook_reference(patch, loaded_model, target_class):
    """Procedimiento original con hook y backward, solo como referencia."""
    holder = {}

    def capture(module, inputs, output):
        holder["activation"] = output
        output.retain_grad()

    model = loaded_model.model
    handle = model.layer2[-1].register_forward_hook(capture)
    try:
        tensor = torch.from_numpy(preprocess_spectral_patch(patch)).unsqueeze(0)
        logits = model(tensor)
        logits[0, target_class].backward()
        activation = holder["activation"]
        weights = activation.grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(64, 64), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().numpy()
        low, high = float(cam.min()), float(cam.max())
        return (cam - low) / (high - low) if high > low else np.zeros_like(cam)
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


@pytest.mark.parametrize("target_class", [0, 1])
def test_matches_notebook_algorithm_and_normal_prediction(loaded_model, patch, target_class):
    original = patch.data.copy()
    prediction = predict_spectral_patch(patch, loaded_model)
    result = calculate_gradcam(patch, loaded_model, target_class=target_class)
    expected = notebook_reference(patch, loaded_model, target_class)
    np.testing.assert_allclose(result.heatmap, expected, atol=1e-6)
    np.testing.assert_equal(patch.data, original)
    assert result.target_probability == pytest.approx(prediction.probabilities[target_class], abs=1e-7)
    assert result.target_class == target_class
    assert result.target_layer == "layer2.1"
    assert result.native_shape == (8, 8)
    assert result.heatmap.shape == (64, 64)
    assert result.heatmap.dtype == np.float32
    assert np.isfinite(result.heatmap).all()
    assert 0 <= result.heatmap.min() <= result.heatmap.max() <= 1


def test_preserves_weights_buffers_gradients_and_prediction(loaded_model, patch):
    model = loaded_model.model
    first_parameter = next(model.parameters())
    first_parameter.grad = torch.ones_like(first_parameter)
    state = {name: value.clone() for name, value in model.state_dict().items()}
    before = predict_spectral_patch(patch, loaded_model)
    with torch.inference_mode():
        first = calculate_gradcam(patch, loaded_model, target_class=1)
    second = calculate_gradcam(patch, loaded_model, target_class=1)
    assert before == predict_spectral_patch(patch, loaded_model)
    np.testing.assert_array_equal(first.heatmap, second.heatmap)
    for name, value in model.state_dict().items():
        assert torch.equal(value, state[name])
    assert torch.equal(first_parameter.grad, torch.ones_like(first_parameter))
    assert all(parameter.grad is None for parameter in list(model.parameters())[1:])
    assert all(not module.training and not module._forward_hooks for module in model.modules())


def test_constant_map_with_frozen_parameters_is_zero(loaded_model, patch):
    with torch.no_grad():
        loaded_model.model.fc.weight.zero_()
    loaded_model.model.requires_grad_(False)
    result = calculate_gradcam(patch, loaded_model, target_class=0)
    assert np.count_nonzero(result.heatmap) == 0
    assert all(not parameter.requires_grad for parameter in loaded_model.model.parameters())


@pytest.mark.parametrize("target_class", [-1, 2, True, 0.5])
def test_rejects_invalid_target_class(loaded_model, patch, target_class):
    with pytest.raises(SpectralGradCAMError, match="clase objetivo"):
        calculate_gradcam(patch, loaded_model, target_class=target_class)


def test_rejects_training_mode_without_changing_it(loaded_model, patch):
    loaded_model.model.train()
    with pytest.raises(SpectralGradCAMError, match="evaluación"):
        calculate_gradcam(patch, loaded_model, target_class=1)
    assert loaded_model.model.training


def test_failure_leaves_no_hooks_or_gradients(loaded_model, patch, monkeypatch):
    def invalid_head(features):
        return features.mean(dim=(2, 3))[:, :2] * float("nan")

    with monkeypatch.context() as context:
        context.setattr(loaded_model.model, "forward_head", invalid_head)
        with pytest.raises(SpectralGradCAMError, match="logits inválidos"):
            calculate_gradcam(patch, loaded_model, target_class=1)
    assert all(not module._forward_hooks for module in loaded_model.model.modules())
    assert all(parameter.grad is None for parameter in loaded_model.model.parameters())
    assert np.isfinite(calculate_gradcam(patch, loaded_model, target_class=1).heatmap).all()


def test_shared_model_concurrent_calls_match_serial_results(loaded_model, patch):
    expected = [calculate_gradcam(patch, loaded_model, target_class=c) for c in (0, 1)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(calculate_gradcam, patch, loaded_model, target_class=c) for c in (0, 1)]
        results = [future.result() for future in futures]
    for actual, reference in zip(results, expected):
        np.testing.assert_allclose(actual.heatmap, reference.heatmap, atol=1e-6)
        assert actual.target_probability == reference.target_probability


def test_real_checkpoint_and_nan_patch():
    contract = load_inference_contract()
    checkpoint = DEFAULT_INFERENCE_CONTRACT_PATH.parent / contract["model"]["checkpoint_relative_path"]
    if not checkpoint.is_file():
        pytest.skip("El checkpoint local no está disponible.")
    try:
        path = resolve_tif_path("model8__test__neg__P_2018_NEG_00015.tif")
    except FileNotFoundError:
        pytest.skip("El GeoTIFF local no está disponible.")
    loaded = load_spectral_model()
    sample = load_spectral_patch(path)
    assert np.isnan(sample.data).any()
    before = predict_spectral_patch(sample, loaded)
    result = calculate_gradcam(sample, loaded, target_class=before.predicted_class)
    np.testing.assert_allclose(result.heatmap, notebook_reference(sample, loaded, before.predicted_class), atol=1e-6)
    assert result.target_probability == pytest.approx(before.probabilities[before.predicted_class], abs=1e-7)
    assert before == predict_spectral_patch(sample, loaded)
