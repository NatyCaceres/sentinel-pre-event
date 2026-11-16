"""Entrada de Community Cloud: prepara archivos y ejecuta el MVP existente."""

import logging
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st
from src.data.product_catalog import calculate_sha256
from src.deployment.assets import CloudAssetError, MANIFEST_PATH, ensure_cloud_assets


@st.cache_resource(show_spinner=False)
def prepare_assets(manifest_hash, url):
    ensure_cloud_assets(url)
    return manifest_hash


url = os.getenv("CLOUD_BUNDLE_URL", "")
if not url:
    try:
        url = st.secrets.get("CLOUD_BUNDLE_URL", "")
    except FileNotFoundError:
        pass

try:
    with st.spinner("Preparando los datos del prototipo..."):
        prepare_assets(calculate_sha256(MANIFEST_PATH), url)
except (CloudAssetError, OSError, ValueError) as error:
    logging.getLogger(__name__).error("Preparación de archivos del MVP: %s", error)
    st.error("No fue posible preparar los archivos del prototipo. Contacta a quien administra la aplicación.")
    st.stop()

runpy.run_path(str(ROOT / "app/app.py"), run_name="__main__")
