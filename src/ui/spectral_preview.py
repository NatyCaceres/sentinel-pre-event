"""Vista opcional RGB y Grad-CAM, con resultados limitados a la sesión."""

import hashlib

import numpy as np
import streamlit as st

from src.spectral.gradcam import calculate_gradcam
from src.spectral.inference import SpectralInferenceError
from src.spectral.visualization import (
    SpectralVisualizationError,
    colorize_gradcam,
    prepare_gradcam_overlay,
    prepare_rgb,
)


def render_spectral_preview(patch, patch_id, prediction_state, load_model, *, error=None):
    """Renderizar después de procesar el botón de inferencia, aun en un placeholder.

Solo calcula Grad-CAM ante un clic explícito. Guarda un único resultado,
identificado por muestra, datos, checkpoint, clase y predicción.
"""
    with st.expander("Ver imagen RGB y explicación visual del parche"):
        if patch is None:
            st.session_state.pop("spectral_gradcam", None)
            st.error("No fue posible cargar la imagen del parche seleccionado.")
            st.caption(str(error))
            return

        current = prediction_state if prediction_state and prediction_state["patch_id"] == patch_id else None
        prediction = current["prediction"] if current else None
        model_hash = current.get("model_sha256") if current else None
        identity = None
        if prediction is not None and model_hash:
            identity = (
                patch_id, hashlib.sha256(patch.data.tobytes()).hexdigest(),
                model_hash, prediction.predicted_class,
                prediction.probabilities, prediction.threshold,
            )
        saved = st.session_state.get("spectral_gradcam")
        if saved is not None and saved["identity"] != identity:
            st.session_state.pop("spectral_gradcam", None)
            saved = None

        try:
            rgb = prepare_rgb(patch)
            if st.button("Generar explicación visual", key="generate_gradcam", disabled=identity is None):
                st.session_state.pop("spectral_gradcam", None)
                saved = None
                with st.spinner("Calculando Grad-CAM para esta muestra..."):
                    loaded = load_model()
                    if loaded.contract["model"]["checkpoint_sha256"] != model_hash:
                        raise SpectralInferenceError("El modelo cambió. Vuelve a analizar la muestra.")
                    result = calculate_gradcam(patch, loaded, target_class=prediction.predicted_class)
                    if abs(result.target_probability - prediction.probabilities[result.target_class]) > 1e-6:
                        raise SpectralInferenceError("La explicación no coincide con la inferencia. Vuelve a analizar la muestra.")
                saved = {"identity": identity, "result": result}
                st.session_state["spectral_gradcam"] = saved

            if saved is None:
                image_column, note_column = st.columns([1, 2], gap="large", vertical_alignment="center")
                with image_column:
                    st.image(rgb, width=256)
                with note_column:
                    st.caption("VISUALIZACIÓN COMPLEMENTARIA")
                    st.subheader("RGB natural")
                    st.write("Composición Sentinel-2 B4/B3/B2.")
                    st.caption("El contraste percentil 2–98 se aplica solo para visualización y no forma parte del preprocesamiento del modelo.")
                    if identity is None:
                        st.caption("Primero pulsa «Analizar patrón espectral» para habilitar la explicación visual.")
                return

            result = saved["result"]
            label = "Con detección posterior" if result.target_class == 1 else "Sin detección cercana"
            st.markdown(f"**Regiones destacadas para la clase {result.target_class} · {label}**")
            st.caption(f"Muestra: {patch_id} · Se explica la clase asignada por el modelo, no la etiqueta observada.")
            st.caption(
                "Esta visualización destaca regiones del parche asociadas a la "
                "puntuación de la clase explicada. Los colores claros indican "
                "mayor intensidad relativa dentro de esta muestra."
            )
            heatmap = colorize_gradcam(result.heatmap)
            overlay = prepare_gradcam_overlay(rgb, result.heatmap)
            for column, title, image in zip(
                st.columns(3, gap="medium"),
                ("RGB de referencia", "Grad-CAM", "Grad-CAM sobre RGB"),
                (rgb, heatmap, overlay),
            ):
                with column:
                    st.caption(title)
                    st.image(image, width=256)
            scale = np.tile(np.linspace(0, 1, 256, dtype=np.float32), (8, 1))
            st.image(colorize_gradcam(scale), width=256)
            st.caption("0 · Menor intensidad — 1 · Mayor intensidad relativa dentro de este parche.")
            if not np.any(result.heatmap):
                st.info("Este cálculo no muestra contraste espacial: el mapa es constante. No implica ausencia de detecciones ni de riesgo.")
            st.caption("Grad-CAM se calcula sobre los ocho canales; el RGB es una referencia visual. No representa probabilidad de incendio ni delimita una superficie quemada.")
            with st.expander("Detalles técnicos de Grad-CAM"):
                st.caption(f"Capa {result.target_layer} · Resolución original {result.native_shape[0]} × {result.native_shape[1]}, ampliada a 64 × 64. La escala se normaliza por parche y no permite comparar intensidades absolutas entre muestras.")
                if np.isnan(patch.data).any():
                    st.caption("Este parche contiene valores faltantes, imputados según el mismo contrato de inferencia del modelo.")
        except (SpectralInferenceError, SpectralVisualizationError, FileNotFoundError, RuntimeError) as exception:
            st.session_state.pop("spectral_gradcam", None)
            st.error("No fue posible generar la explicación visual. La clasificación se conserva.")
            st.caption(str(exception))
