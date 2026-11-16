from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.spectral.gradcam import SpectralGradCAM
from src.spectral.inference import SpectralPrediction
from src.spectral.patch import SpectralPatch
from src.spectral.visualization import (
    SpectralVisualizationError,
    colorize_gradcam,
    prepare_gradcam_overlay,
)


def preview_app(patch, state, loader):
    import streamlit as st
    from src.ui.spectral_preview import render_spectral_preview

    if "sample" not in st.session_state:
        st.session_state["sample"] = patch
        st.session_state["sample_id"] = "sample-A"
        st.session_state["prediction"] = state
    # Misma secuencia del MVP: reservar posición, procesar inferencia, renderizar.
    preview = st.container(key="rgb_preview_panel")
    if st.button("Analizar", key="analyze"):
        st.session_state["prediction"] = state
        st.session_state.pop("spectral_gradcam", None)
    with preview:
        render_spectral_preview(
            st.session_state["sample"], st.session_state["sample_id"],
            st.session_state["prediction"], loader,
        )


@pytest.fixture
def preview(monkeypatch):
    from src.ui import spectral_preview

    # La clase observada no se entrega al componente: explica la predicción.
    # p1=0.497 decide clase 1 por umbral 0.495, aunque no sea el argmax.
    prediction = SpectralPrediction((0.503, 0.497), 0.497, 1, 0.495)
    state = {"patch_id": "sample-A", "prediction": prediction, "model_sha256": "hash-A"}
    patch = SpectralPatch(
        data=np.random.default_rng(0).random((8, 64, 64), dtype=np.float32),
        channels=("B2", "B3", "B4", "B8", "B11", "B12", "NDMI", "NBR"),
        path=Path("synthetic.tif"),
    )
    loader = Mock(return_value=SimpleNamespace(contract={"model": {"checkpoint_sha256": "hash-A"}}))
    calculate = Mock(return_value=SpectralGradCAM(
        heatmap=np.tile(np.linspace(0, 1, 64, dtype=np.float32), (64, 1)),
        target_class=1, target_probability=0.497, native_shape=(8, 8),
    ))
    monkeypatch.setattr(spectral_preview, "calculate_gradcam", calculate)
    app = AppTest.from_function(preview_app, args=(patch, state, loader), default_timeout=15)
    return app, loader, calculate


def test_generate_only_on_click_and_reuse_during_reruns(preview):
    app, loader, calculate = preview
    app.run()
    assert not app.exception
    loader.assert_not_called()
    calculate.assert_not_called()
    app.button(key="generate_gradcam").click().run()
    assert not app.exception
    assert not app.error
    calculate.assert_called_once()
    assert calculate.call_args.kwargs["target_class"] == 1
    assert len(app.image) == 4  # RGB, mapa, superposición y escala.
    app.run()
    calculate.assert_called_once()
    loader.assert_called_once()


def test_analysis_enables_explanation_in_same_run(preview):
    app, loader, calculate = preview
    app.run()
    app.session_state["prediction"] = None
    app.run()
    assert app.button(key="generate_gradcam").disabled
    app.button(key="analyze").click().run()
    assert not app.exception
    assert not app.button(key="generate_gradcam").disabled
    calculate.assert_not_called()


@pytest.mark.parametrize("change", ["selection", "data", "model", "prediction"])
def test_discards_obsolete_explanation(preview, change):
    app, loader, calculate = preview
    app.run().button(key="generate_gradcam").click().run()
    if change == "selection":
        app.session_state["sample_id"] = "sample-B"
    elif change == "data":
        app.session_state["sample"].data[0, 0, 0] = 0.123
    else:
        state = dict(app.session_state["prediction"])
        if change == "model":
            state["model_sha256"] = "hash-B"
        else:
            state["prediction"] = SpectralPrediction((0.8, 0.2), 0.2, 0, 0.495)
        app.session_state["prediction"] = state
    app.run()
    assert not app.exception
    assert "spectral_gradcam" not in app.session_state
    assert len(app.image) == 1
    calculate.assert_called_once()


def test_calculation_failure_keeps_classification(preview):
    app, loader, calculate = preview
    app.run()
    before = app.session_state["prediction"]["prediction"]
    calculate.side_effect = RuntimeError("Error de prueba")
    app.button(key="generate_gradcam").click().run()
    assert not app.exception
    assert "La clasificación se conserva" in app.error[0].value
    assert app.session_state["prediction"]["prediction"] == before
    assert "spectral_gradcam" not in app.session_state


def test_overlay_preserves_inputs_and_opacity_endpoints():
    rgb = np.full((64, 64, 3), 0.25, dtype=np.float32)
    cam = np.tile(np.linspace(0, 1, 64, dtype=np.float32), (64, 1))
    original = cam.copy()
    colors = colorize_gradcam(cam)
    assert colors.dtype == np.float32
    assert np.all((colors >= 0) & (colors <= 1))
    np.testing.assert_array_equal(prepare_gradcam_overlay(rgb, cam, alpha=0), rgb)
    np.testing.assert_array_equal(prepare_gradcam_overlay(rgb, cam, alpha=1), colors)
    np.testing.assert_array_equal(cam, original)
    assert np.all(rgb == 0.25)


@pytest.mark.parametrize("cam", [np.full((2, 2), np.nan), np.full((2, 2), 1.1), np.zeros(3)])
def test_invalid_heatmap_is_not_silently_colored(cam):
    with pytest.raises(SpectralVisualizationError):
        colorize_gradcam(cam)
