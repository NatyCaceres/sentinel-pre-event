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
   BARRA LATERAL
   ========================================================== */

section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E1E5EA;
}

section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
    gap: 0.55rem;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    min-height: 2.6rem;
}

section[data-testid="stSidebar"] details {
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

/* La selección mantiene continuidad visual entre el mapa
   y la ficha técnica sin añadir otra tarjeta de métricas. */
div[class*="st-key-explorer_selection_panel_"] {
    background: linear-gradient(120deg, #FFFFFF 0%, #F3F7FC 100%);
    border-left: 4px solid #2B61AD;
    border-radius: 17px;
    box-shadow: 0 12px 32px rgba(25, 45, 75, 0.09);
    margin-top: 0.65rem;
    padding: 1rem 1.2rem;
}

.active-sample__eyebrow {
    color: #3265A4;
    font-size: 0.68rem;
    font-weight: 750;
    letter-spacing: 0.075em;
    margin-bottom: 0.38rem;
}

.active-sample__content {
    display: block;
}

.active-sample__id {
    color: #17263A;
    font-size: 1.22rem;
    font-weight: 720;
    overflow-wrap: anywhere;
}

.active-sample__place {
    color: #69778B;
    font-size: 0.82rem;
    margin-top: 0.24rem;
}

.active-sample__class {
    align-items: center;
    background-color: #FFFFFF;
    border: 1px solid #DCE4EE;
    border-radius: 999px;
    color: #27374B;
    display: inline-flex;
    font-size: 0.78rem;
    font-weight: 650;
    margin-top: 0.65rem;
    padding: 0.48rem 0.72rem;
}


/* ==========================================================
   VISTA RGB COMPLEMENTARIA
   ========================================================== */

.st-key-rgb_preview_panel details {
    background-color: #FFFFFF;
    border: 0;
    border-left: 4px solid #2B61AD;
    border-radius: 16px;
    box-shadow: 0 10px 28px rgba(25, 45, 75, 0.08);
    overflow: hidden;
}

.st-key-rgb_preview_panel details summary {
    background: linear-gradient(120deg, #F8FAFD 0%, #EDF3FA 100%);
    color: #24364C;
    font-weight: 650;
    padding-bottom: 0.9rem;
    padding-top: 0.9rem;
}

.st-key-rgb_preview_panel details[open] summary {
    border-bottom: 1px solid #E1E8F1;
}

.st-key-rgb_preview_panel img {
    border-radius: 12px;
    box-shadow: 0 8px 22px rgba(25, 45, 75, 0.12);
}

/* ==========================================================
   FICHA DE LA MUESTRA
   ========================================================== */

.sample-profile {
    background-color: #FFFFFF;
    border-radius: 20px;
    box-shadow: 0 18px 45px rgba(25, 45, 75, 0.11);
    margin: 1rem 0 1.3rem;
    overflow: hidden;
}

.sample-profile__hero {
    align-items: center;
    background: linear-gradient(120deg, #102C4C 0%, #194C82 100%);
    color: #FFFFFF;
    display: flex;
    gap: 2rem;
    justify-content: space-between;
    padding: 1.55rem 1.8rem;
}

.sample-profile__eyebrow,
.prediction-intro__eyebrow,
.prediction-result__eyebrow {
    font-size: 0.72rem;
    font-weight: 750;
    letter-spacing: 0.075em;
}

.sample-profile__eyebrow {
    color: #BFD7F2;
}

.sample-profile__id {
    font-size: clamp(1.35rem, 2vw, 1.85rem);
    font-weight: 720;
    letter-spacing: -0.025em;
    line-height: 1.2;
    margin-top: 0.35rem;
    overflow-wrap: anywhere;
}

.sample-profile__place {
    color: #D9E7F5;
    font-size: 0.92rem;
    margin-top: 0.45rem;
}

.observed-class {
    background-color: rgba(255, 255, 255, 0.10);
    border: 1px solid rgba(255, 255, 255, 0.22);
    border-radius: 14px;
    display: flex;
    flex: 0 0 min(24rem, 38%);
    flex-direction: column;
    gap: 0.45rem;
    padding: 0.95rem 1.1rem;
}

.observed-class--positive {
    border-left: 4px solid #F05468;
}

.observed-class--negative {
    border-left: 4px solid #5B91E5;
}

.observed-class__label {
    color: #BFD7F2;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.065em;
}

.observed-class__value {
    font-size: 0.94rem;
    font-weight: 680;
}

.sample-profile__facts {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    padding: 0.45rem 1.8rem 0.2rem;
}

.sample-profile__facts > div {
    border-bottom: 1px solid #E8EDF3;
    padding: 0.9rem 1.2rem 0.9rem 0;
}

.sample-profile__facts > div:nth-last-child(-n + 3) {
    border-bottom: 0;
}

.sample-profile__facts span,
.sample-profile__facts strong {
    display: block;
}

.sample-profile__facts span {
    color: #748094;
    font-size: 0.77rem;
    margin-bottom: 0.3rem;
}

.sample-profile__facts strong {
    color: #17263A;
    font-size: 0.91rem;
    font-weight: 680;
}

.sample-profile__note {
    background-color: #EDF4FC;
    color: #315A87;
    font-size: 0.78rem;
    padding: 0.7rem 1.8rem;
}


/* ==========================================================
   PANEL DE PREDICCIÓN
   ========================================================== */

.st-key-prediction_panel {
    background-color: #FFFFFF;
    border: 1px solid #DCE4EE;
    border-radius: 20px;
    box-shadow: 0 18px 45px rgba(25, 45, 75, 0.10);
    margin-top: 0.9rem;
    padding: 1.45rem 1.6rem 1.35rem;
}

.prediction-intro {
    align-items: center;
    display: flex;
    gap: 1rem;
    margin-bottom: 1.15rem;
}

.prediction-intro__mark {
    align-items: center;
    background: linear-gradient(135deg, #173F77, #2D68B5);
    border-radius: 14px;
    color: #FFFFFF;
    display: flex;
    flex: 0 0 3.2rem;
    font-size: 0.82rem;
    font-weight: 800;
    height: 3.2rem;
    justify-content: center;
    letter-spacing: 0.06em;
}

.prediction-intro__eyebrow {
    color: #3265A4;
}

.prediction-intro__title {
    color: #17263A;
    font-size: 1.38rem;
    font-weight: 720;
    line-height: 1.2;
    margin-top: 0.2rem;
}

.prediction-intro__copy {
    color: #69778B;
    font-size: 0.86rem;
    margin-top: 0.32rem;
}

.prediction-result {
    background-color: #F4F7FB;
    border-left: 4px solid #5B91E5;
    border-radius: 14px;
    margin-top: 1.2rem;
    padding: 1.1rem 1.2rem 1.2rem;
}

.prediction-result--positive {
    border-left-color: #DB3E57;
}

.prediction-result__eyebrow {
    color: #68768A;
}

.prediction-result__title {
    color: #17263A;
    font-size: 1.18rem;
    font-weight: 720;
    margin: 0.25rem 0 1rem;
}

.probability-row + .probability-row {
    margin-top: 0.85rem;
}

.probability-row__label {
    color: #4D5B6E;
    display: flex;
    font-size: 0.8rem;
    justify-content: space-between;
    margin-bottom: 0.35rem;
}

.probability-row__label strong {
    color: #17263A;
}

.probability-track {
    background-color: #DFE6EF;
    border-radius: 999px;
    height: 0.55rem;
    overflow: hidden;
}

.probability-fill {
    border-radius: inherit;
    display: block;
    height: 100%;
}

.probability-fill--negative {
    background: linear-gradient(90deg, #366FBE, #6CA1E9);
}

.probability-fill--positive {
    background: linear-gradient(90deg, #C92F49, #F05C70);
}

.scope-note {
    align-items: flex-start;
    background-color: #EEF4FA;
    border-radius: 12px;
    color: #536276;
    display: flex;
    flex-direction: column;
    font-size: 0.82rem;
    gap: 0.28rem;
    margin-top: 0.9rem;
    padding: 0.85rem 1rem;
}

.scope-note strong {
    color: #234F7E;
    font-size: 0.76rem;
    letter-spacing: 0.035em;
    text-transform: uppercase;
}

@media (max-width: 800px) {
    .sample-profile__hero {
        align-items: stretch;
        flex-direction: column;
    }

    .observed-class {
        flex-basis: auto;
    }

    .sample-profile__facts {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .sample-profile__facts > div:nth-last-child(-n + 3) {
        border-bottom: 1px solid #E8EDF3;
    }

    .sample-profile__facts > div:nth-last-child(-n + 2) {
        border-bottom: 0;
    }

}


/* ==========================================================
   MAPA
   ========================================================== */

.map-context-bar {
    align-items: center;
    background-color: #EDF3FA;
    border-radius: 12px;
    color: #58677A;
    display: flex;
    font-size: 0.78rem;
    gap: 1.2rem;
    justify-content: space-between;
    margin: 0.75rem 0 0.85rem;
    padding: 0.72rem 0.9rem;
}

.map-context-bar__count,
.map-context-bar__legend,
.map-context-bar__legend span {
    align-items: center;
    display: flex;
}

.map-context-bar__count {
    flex-wrap: wrap;
    gap: 0.35rem;
}

.map-context-bar__count strong {
    color: #22344A;
}

.map-context-bar__legend {
    flex-wrap: wrap;
    gap: 0.8rem;
    justify-content: flex-end;
}

.map-context-bar__legend span {
    color: #34445A;
    gap: 0.32rem;
    white-space: nowrap;
}

.legend-dot {
    border-radius: 999px;
    display: inline-block;
    height: 0.62rem;
    width: 0.62rem;
}

.legend-dot--positive {
    background-color: #DF3E57;
}

.legend-dot--negative {
    background-color: #437CCB;
}

.legend-dot--cluster {
    background-color: #303747;
}

iframe {
    border-radius: 14px;
}

@media (max-width: 1000px) {
    .map-context-bar {
        align-items: flex-start;
        flex-direction: column;
    }

    .map-context-bar__legend {
        justify-content: flex-start;
    }
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
