from __future__ import annotations

import streamlit as st


APP_CSS = """
<style>

/* ==========================================================
   BASE
   ========================================================== */

.stApp {
    background-color: #F7F8FA;
}

.block-container {
    max-width: 1520px;
    padding-top: 1.4rem;
    padding-bottom: 2.5rem;
}


/* ==========================================================
   PANEL DE FILTROS
   ========================================================== */

div[data-testid="stColumn"]:has(.filters-sticky-anchor),
div[data-testid="column"]:has(.filters-sticky-anchor) {
    position: sticky;
    top: 1.25rem;
    align-self: flex-start;
    height: fit-content;
}


/* Tarjeta principal de consulta */
div[data-testid="stVerticalBlockBorderWrapper"]:has(
    .filters-panel-anchor
) {
    background-color: #FFFFFF;
    border: 1px solid #E1E5EA;
    border-radius: 16px;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.045);
}


/* Reducir espacio vertical dentro del panel */
div[data-testid="stVerticalBlockBorderWrapper"]:has(
    .filters-panel-anchor
) div[data-testid="stVerticalBlock"] {
    gap: 0.55rem;
}


/* Inputs ligeramente más compactos */
div[data-testid="stVerticalBlockBorderWrapper"]:has(
    .filters-panel-anchor
) div[data-baseweb="select"] > div {
    min-height: 2.6rem;
}


/* Expander de filtros secundarios */
div[data-testid="stVerticalBlockBorderWrapper"]:has(
    .filters-panel-anchor
) details {
    background-color: #F8FAFC;
    border: 1px solid #E7EAF0;
    border-radius: 10px;
}

/* ==========================================================
   HEADER
   ========================================================== */

.product-name {
    color: #2457A6;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.055em;
    margin-bottom: 0.25rem;
}

.product-title {
    color: #1F2937;
    font-size: 1.85rem;
    font-weight: 730;
    line-height: 1.18;
    margin: 0;
}

.product-subtitle {
    color: #667085;
    font-size: 0.94rem;
    margin-top: 0.45rem;
    max-width: 900px;
}

.validated-status {
    color: #667085;
    font-size: 0.85rem;
    text-align: right;
}


/* ==========================================================
   MÉTRICAS
   ========================================================== */

div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E3E7EC;
    border-radius: 15px;
    padding: 0.85rem 1rem;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.035);
}

div[data-testid="stMetricLabel"] {
    color: #667085;
    font-size: 0.80rem;
}

div[data-testid="stMetricValue"] {
    color: #172033;
}


/* ==========================================================
   INPUTS
   ========================================================== */

div[data-baseweb="select"] > div {
    border-radius: 10px;
}

div[data-testid="stDateInput"] input {
    border-radius: 10px;
}

div[data-testid="stFormSubmitButton"] button {
    min-height: 2.8rem;
    border-radius: 10px;
    font-weight: 650;
}


/* ==========================================================
   NAVEGACIÓN
   ========================================================== */

div[data-testid="stSegmentedControl"] {
    margin-top: 0.15rem;
    margin-bottom: 0.4rem;
}


/* ==========================================================
   CONTENEDORES
   ========================================================== */

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px;
}


/* ==========================================================
   MAPA
   ========================================================== */

iframe {
    border-radius: 14px;
}


/* ==========================================================
   TEXTO SECUNDARIO
   ========================================================== */

.muted-note {
    color: #667085;
    font-size: 0.83rem;
}

.section-kicker {
    color: #2457A6;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.055em;
}


/* ==========================================================
   SEPARADORES
   ========================================================== */

hr {
    border-color: #E3E7EC;
}

</style>
"""


def apply_app_styles() -> None:
    """
    Aplica la identidad visual principal del MVP.
    """

    st.markdown(
        APP_CSS,
        unsafe_allow_html=True,
    )