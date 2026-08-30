from __future__ import annotations

import streamlit as st
from streamlit_folium import st_folium

from src.data.phase1 import Phase1Dataset
from src.data.phase1_geographic import (
    load_phase1_geographic,
)
from src.geo.folium_map import (
    build_phase1_folium_map,
)
from src.geo.phase1_map import (
    build_map_points,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Análisis espectral pre-evento",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# ESTILOS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #F7F8FA;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2.3rem;
        padding-bottom: 3rem;
    }

    .project-badge {
        display: inline-block;
        background-color: #E8F0FE;
        color: #2457A6;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-bottom: 0.8rem;
    }

    .main-title {
        color: #1F2937;
        font-size: 2.35rem;
        font-weight: 750;
        line-height: 1.15;
        margin-bottom: 0;
    }

    .main-description {
        color: #667085;
        font-size: 1rem;
        margin-top: 0.65rem;
        margin-bottom: 1.8rem;
        max-width: 900px;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 1rem 1.15rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }

    div[data-testid="stMetricLabel"] {
        color: #667085;
    }

    div[data-testid="stMetricValue"] {
        color: #1F2937;
    }

    .section-label {
        color: #2457A6;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        margin-top: 1.8rem;
        margin-bottom: 0.25rem;
    }

    .section-title {
        color: #1F2937;
        font-size: 1.55rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .section-description {
        color: #667085;
        margin-bottom: 1rem;
    }

    .filter-title {
        color: #475467;
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .map-help {
        color: #667085;
        font-size: 0.84rem;
        margin-top: 0.1rem;
        margin-bottom: 0.6rem;
    }

    .small-note {
        color: #98A2B3;
        font-size: 0.80rem;
        margin-top: 0.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CARGA DE DATOS
# ============================================================

@st.cache_resource
def load_phase1_dataset() -> Phase1Dataset:
    return Phase1Dataset.load()


@st.cache_data
def load_geographic_data():
    return load_phase1_geographic()


dataset = load_phase1_dataset()
geographic = load_geographic_data()


# ============================================================
# PREPARACIÓN GEOGRÁFICA
# ============================================================

mapped_data = geographic[
    geographic["administrative_status"]
    == "assigned"
].copy()

all_points = build_map_points(
    mapped_data
)

summary = dataset.summary()


# ============================================================
# ENCABEZADO
# ============================================================

st.markdown(
    '<div class="project-badge">'
    'EXPLORADOR GEOESPACIAL · FASE 1'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">'
    'Análisis espectral del terreno previo al fuego'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-description">'
    'Exploración geográfica de las muestras pre-evento '
    'construidas para el análisis con Sentinel-2 y deep learning.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# MÉTRICAS GENERALES
# ============================================================

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Parches del estudio",
    f"{summary['unique_patches']:,}".replace(",", "."),
)

metric_2.metric(
    "Grupos / eventos",
    f"{summary['unique_groups']:,}".replace(",", "."),
)

metric_3.metric(
    "Temporadas analizadas",
    len(
        geographic["season_year"]
        .dropna()
        .unique()
    ),
)


# ============================================================
# EXPLORACIÓN GEOGRÁFICA
# ============================================================

st.markdown(
    '<div class="section-label">'
    'DISTRIBUCIÓN DE LAS MUESTRAS'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">'
    'Exploración geográfica'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'Explora la distribución espacial de los parches y '
    'filtra las muestras según su asociación con detecciones '
    'posteriores de fuego activo.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# FILTRO DE CLASE
# ============================================================

st.markdown(
    '<div class="filter-title">'
    'Clase de la muestra'
    '</div>',
    unsafe_allow_html=True,
)

class_filter = st.segmented_control(
    "Clase de la muestra",
    options=[
        "Todas",
        "Con detección posterior",
        "Sin detección cercana",
    ],
    default="Todas",
    label_visibility="collapsed",
    key="class_filter",
)

if class_filter is None:
    class_filter = "Todas"


# ============================================================
# APLICAR FILTRO
# ============================================================

if class_filter == "Con detección posterior":

    points = all_points[
        all_points["etiqueta"] == 1
    ].copy()

elif class_filter == "Sin detección cercana":

    points = all_points[
        all_points["etiqueta"] == 0
    ].copy()

else:

    points = all_points.copy()


# ============================================================
# MÉTRICAS DINÁMICAS
# ============================================================

visible_positive = int(
    (points["etiqueta"] == 1).sum()
)

visible_negative = int(
    (points["etiqueta"] == 0).sum()
)

visible_groups = int(
    points["group_id"].nunique()
)


info_1, info_2, info_3 = st.columns(3)

info_1.metric(
    "Muestras visibles",
    f"{len(points):,}".replace(",", "."),
)

info_2.metric(
    "Grupos visibles",
    f"{visible_groups:,}".replace(",", "."),
)

if class_filter == "Todas":

    info_3.metric(
        "Con / Sin detección",
        f"{visible_positive} / {visible_negative}",
        help=(
            "Con detección posterior / "
            "Sin detección cercana"
        ),
    )

else:

    info_3.metric(
        "Clase visible",
        class_filter,
    )


# ============================================================
# LEYENDA
# ============================================================

if class_filter == "Todas":

    legend_1, legend_2, legend_3 = st.columns(
        [1.1, 1.1, 1.8]
    )

    legend_1.markdown(
        "🔴 **Con detección posterior**"
    )

    legend_2.markdown(
        "🔵 **Sin detección cercana**"
    )

    legend_3.markdown(
        "⚫ **Cluster con ambas clases**"
    )

elif class_filter == "Con detección posterior":

    st.markdown(
        "🔴 **Con detección posterior** · "
        "Los clusters visibles contienen únicamente "
        "muestras de esta clase."
    )

else:

    st.markdown(
        "🔵 **Sin detección cercana** · "
        "Los clusters visibles contienen únicamente "
        "muestras de esta clase."
    )


st.markdown(
    '<div class="map-help">'
    'Pasa el cursor sobre un cluster para observar su '
    'cobertura y haz clic para acercarte a las muestras.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# MAPA
# ============================================================

map_object = build_phase1_folium_map(
    points=points,
    class_filter=class_filter,
)

st_folium(
    map_object,
    height=620,
    use_container_width=True,
    returned_objects=[],
    key=f"phase1_map_{class_filter}",
)


# ============================================================
# NOTA
# ============================================================

st.markdown(
    '<div class="small-note">'
    '1 registro se conserva en la metadata de Fase 1, '
    'pero no se representa debido a que no fue posible '
    'asignarle una unidad administrativa con la '
    'DPA 2023 utilizada.'
    '</div>',
    unsafe_allow_html=True,
)