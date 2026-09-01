from __future__ import annotations

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.data.explorer_filters import (
    apply_explorer_filters,
)
from src.data.explorer_selection import (
    get_available_group_ids,
    get_available_patch_ids,
    get_group_patch_ids,
    get_patch,
)
from src.data.product_catalog_loader import (
    load_product_catalog,
)
from src.geo.folium_map import (
    build_phase1_folium_map,
)
from src.geo.phase1_map import (
    build_map_points,
    format_season_label,
)
from src.ui.styles import (
    apply_app_styles,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Sentinel Pre-Event",
    page_icon="🛰️",
    layout="wide",
)

apply_app_styles()


# ============================================================
# HELPERS
# ============================================================

def format_date(value) -> str:
    if pd.isna(value):
        return "Sin información"

    return pd.Timestamp(
        value
    ).strftime("%d-%m-%Y")


def format_percent(value) -> str:
    if pd.isna(value):
        return "Sin información"

    return f"{float(value):.1f} %"


# ============================================================
# CATÁLOGO
# ============================================================

@st.cache_data
def load_catalog():
    return load_product_catalog()


catalog = load_catalog()


# Solo el mapa necesita correspondencia territorial resuelta.
mapped_catalog = catalog[
    catalog["map_eligible"]
].copy()

all_points = build_map_points(
    mapped_catalog
)


# ============================================================
# OPCIONES DE FILTRO
# ============================================================

season_years = sorted(
    all_points["season_year"]
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)

season_labels = {
    format_season_label(year): year
    for year in season_years
}

region_options = sorted(
    all_points["REGION"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

reference_dates = pd.to_datetime(
    all_points["fecha_referencia"],
    errors="coerce",
)

minimum_reference_date = (
    reference_dates
    .min()
    .date()
)

maximum_reference_date = (
    reference_dates
    .max()
    .date()
)


# ============================================================
# LAYOUT PRINCIPAL
# ============================================================

filters_column, content_column = st.columns(
    [0.72, 3.28],
    gap="large",
)


# ============================================================
# PANEL IZQUIERDO
# ============================================================

with filters_column:

    st.markdown(
        '<div class="filters-sticky-anchor"></div>',
        unsafe_allow_html=True,
    )

    with st.container(
        border=True,
    ):

        st.markdown(
            '<div class="filters-panel-anchor"></div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "CONFIGURAR CONSULTA"
        )

        st.markdown(
            "### Nueva consulta"
        )

        st.caption(
            "Ajusta los criterios de exploración."
        )

        with st.form(
            "explorer_query_form",
            border=False,
        ):

            selected_season_label = st.selectbox(
                "Temporada",
                options=[
                    "Todas"
                ] + list(
                    season_labels.keys()
                ),
                index=0,
                key="query_season",
            )

            selected_region = st.selectbox(
                "Región",
                options=[
                    "Todas"
                ] + region_options,
                index=0,
                key="query_region",
            )

            class_filter = st.selectbox(
                "Clase",
                options=[
                    "Todas",
                    "Con detección posterior",
                    "Sin detección cercana",
                ],
                index=0,
                key="query_class",
            )

            # Los controles menos utilizados se mantienen
            # disponibles sin ocupar permanentemente espacio.
            with st.expander(
                "Más filtros",
                expanded=False,
            ):

                selected_dates = st.date_input(
                    "Fecha de referencia",
                    value=(
                        minimum_reference_date,
                        maximum_reference_date,
                    ),
                    min_value=minimum_reference_date,
                    max_value=maximum_reference_date,
                    key="query_dates",
                )

                multipatch_only = st.checkbox(
                    "Solo grupos multipatch",
                    value=False,
                    key="query_multipatch",
                    help=(
                        "Conserva únicamente grupos "
                        "con más de un parche disponible."
                    ),
                )

            st.write("")

            st.form_submit_button(
                "Aplicar filtros",
                type="primary",
                use_container_width=True,
            )
# ============================================================
# TRADUCCIÓN DE FILTROS
# ============================================================

if selected_season_label == "Todas":
    selected_season_year = None
else:
    selected_season_year = season_labels[
        selected_season_label
    ]


if selected_region == "Todas":
    region_filter = None
else:
    region_filter = selected_region


class_value = {
    "Todas": None,
    "Con detección posterior": 1,
    "Sin detección cercana": 0,
}[class_filter]


if isinstance(
    selected_dates,
    (tuple, list),
):

    if len(selected_dates) == 2:
        reference_start = selected_dates[0]
        reference_end = selected_dates[1]

    elif len(selected_dates) == 1:
        reference_start = selected_dates[0]
        reference_end = selected_dates[0]

    else:
        reference_start = minimum_reference_date
        reference_end = maximum_reference_date

else:
    reference_start = minimum_reference_date
    reference_end = maximum_reference_date


# ============================================================
# FILTRADO
# ============================================================

points = apply_explorer_filters(
    all_points,
    season_year=selected_season_year,
    region=region_filter,
    label=class_value,
    reference_start=reference_start,
    reference_end=reference_end,
    multipatch_only=multipatch_only,
)


# ============================================================
# CONTENIDO DERECHO
# ============================================================

with content_column:

    # --------------------------------------------------------
    # APP BAR
    # --------------------------------------------------------

    brand_column, status_column = st.columns(
        [4, 1]
    )

    with brand_column:

        st.markdown(
            '<div class="product-name">'
            '🛰 SENTINEL PRE-EVENT'
            '</div>',
            unsafe_allow_html=True,
        )

    with status_column:

        st.markdown(
            '<div class="validated-status">'
            '🟢 Datos validados<br>'
            'Sentinel-2 · 8 canales'
            '</div>',
            unsafe_allow_html=True,
        )


    # --------------------------------------------------------
    # CABECERA
    # --------------------------------------------------------

    st.markdown(
        '<div class="product-title">'
        'Análisis espectral del terreno previo al fuego'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="product-subtitle">'
        'Explora muestras Sentinel-2 pre-evento, '
        'revisa su contexto territorial y posteriormente '
        'analiza sus características espectrales '
        'mediante deep learning.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write("")


    # --------------------------------------------------------
    # NAVEGACIÓN
    # --------------------------------------------------------

    current_view = st.segmented_control(
        "Navegación",
        options=[
            "01 · Mapa",
            "02 · Parches",
        ],
        default="01 · Mapa",
        label_visibility="collapsed",
        key="main_navigation",
    )

    if current_view is None:
        current_view = "01 · Mapa"


    # --------------------------------------------------------
    # SIN RESULTADOS
    # --------------------------------------------------------

    if points.empty:

        st.warning(
            "No existen muestras que cumplan "
            "la combinación de filtros seleccionada."
        )

        st.stop()


    # --------------------------------------------------------
    # MÉTRICAS
    # --------------------------------------------------------

    visible_positive = int(
        (
            points["etiqueta"] == 1
        ).sum()
    )

    visible_negative = int(
        (
            points["etiqueta"] == 0
        ).sum()
    )

    visible_groups = int(
        points["group_id"].nunique()
    )


    # ========================================================
    # VISTA 01 — MAPA
    # ========================================================

    if current_view == "01 · Mapa":

        st.markdown(
            '<div class="section-kicker">'
            '01 · PANORAMA GEOGRÁFICO'
            '</div>',
            unsafe_allow_html=True,
        )

        st.subheader(
            "Distribución de las muestras"
        )

        st.caption(
            "Lectura espacial del conjunto que cumple "
            "los criterios definidos en la consulta."
        )


        metric_1, metric_2, metric_3, metric_4 = (
            st.columns(4)
        )

        metric_1.metric(
            "Muestras",
            f"{len(points):,}".replace(
                ",",
                ".",
            ),
        )

        metric_2.metric(
            "Grupos",
            f"{visible_groups:,}".replace(
                ",",
                ".",
            ),
        )

        metric_3.metric(
            "Con detección",
            f"{visible_positive:,}".replace(
                ",",
                ".",
            ),
        )

        metric_4.metric(
            "Sin detección",
            f"{visible_negative:,}".replace(
                ",",
                ".",
            ),
        )

        st.write("")


        # ----------------------------------------------------
        # LEYENDA
        # ----------------------------------------------------

        if class_filter == "Todas":

            legend_1, legend_2, legend_3 = (
                st.columns(
                    [1.15, 1.15, 1.7]
                )
            )

            legend_1.markdown(
                "🔴 **Con detección posterior**"
            )

            legend_2.markdown(
                "🔵 **Sin detección cercana**"
            )

            legend_3.markdown(
                "⚫ **Cluster de muestras**"
            )

        elif (
            class_filter
            == "Con detección posterior"
        ):

            st.markdown(
                "🔴 **Con detección posterior**"
            )

        else:

            st.markdown(
                "🔵 **Sin detección cercana**"
            )


        st.caption(
            "Pasa el cursor sobre un cluster para "
            "observar su cobertura y haz clic "
            "para acercarte."
        )


        # ----------------------------------------------------
        # MAPA
        # ----------------------------------------------------

        map_object = build_phase1_folium_map(
            points=points,
            class_filter=class_filter,
        )

        map_key = (
            f"product_map_"
            f"{selected_season_year}_"
            f"{region_filter}_"
            f"{class_filter}_"
            f"{reference_start}_"
            f"{reference_end}_"
            f"{multipatch_only}"
        )

        st_folium(
            map_object,
            height=650,
            use_container_width=True,
            returned_objects=[],
            key=map_key,
        )


    # ========================================================
    # VISTA 02 — PARCHES
    # ========================================================

    else:

        st.markdown(
            '<div class="section-kicker">'
            '02 · EXPLORACIÓN DE PARCHES'
            '</div>',
            unsafe_allow_html=True,
        )

        st.subheader(
            "Revisar una muestra"
        )

        st.caption(
            "Selecciona un parche o un grupo para "
            "consultar sus antecedentes principales."
        )


        # ----------------------------------------------------
        # CATÁLOGO DE LA CONSULTA
        # ----------------------------------------------------

        visible_patch_ids = set(
            points["patch_id"]
            .astype(str)
        )

        filtered_catalog = catalog[
            catalog["patch_id"]
            .astype(str)
            .isin(
                visible_patch_ids
            )
        ].copy()


        # ----------------------------------------------------
        # MODO DE BÚSQUEDA
        # ----------------------------------------------------

        search_mode = st.segmented_control(
            "Buscar por",
            options=[
                "Parche",
                "Grupo",
            ],
            default="Parche",
            key="patch_search_mode",
        )

        if search_mode is None:
            search_mode = "Parche"


        # ----------------------------------------------------
        # PARCHE
        # ----------------------------------------------------

        if search_mode == "Parche":

            patch_options = (
                get_available_patch_ids(
                    filtered_catalog
                )
            )

            selected_patch_id = st.selectbox(
                "Patch ID",
                options=patch_options,
                key="selected_patch",
            )


        # ----------------------------------------------------
        # GRUPO
        # ----------------------------------------------------

        else:

            group_options = (
                get_available_group_ids(
                    filtered_catalog
                )
            )

            selected_group_id = st.selectbox(
                "Grupo",
                options=group_options,
                key="selected_group",
            )

            group_patch_options = (
                get_group_patch_ids(
                    filtered_catalog,
                    selected_group_id,
                )
            )

            selected_patch_id = st.selectbox(
                "Parche del grupo",
                options=group_patch_options,
                key="selected_group_patch",
            )


        # ----------------------------------------------------
        # CARGAR PATCH
        # ----------------------------------------------------

        selected_patch = get_patch(
            filtered_catalog,
            selected_patch_id,
        )


        class_label = (
            "Con detección posterior"
            if int(
                selected_patch["etiqueta"]
            ) == 1
            else "Sin detección cercana"
        )

        class_icon = (
            "🔴"
            if int(
                selected_patch["etiqueta"]
            ) == 1
            else "🔵"
        )

        season_label = (
            format_season_label(
                int(
                    selected_patch[
                        "season_year"
                    ]
                )
            )
        )


        # ----------------------------------------------------
        # FICHA DEL PATCH
        # ----------------------------------------------------

        with st.container(
            border=True,
        ):

            title_column, class_column = (
                st.columns(
                    [3, 1]
                )
            )

            with title_column:

                st.caption(
                    "PARCHE SELECCIONADO"
                )

                st.subheader(
                    str(
                        selected_patch[
                            "patch_id"
                        ]
                    )
                )

                if pd.notna(
                    selected_patch["COMUNA"]
                ):

                    st.write(
                        f"{selected_patch['COMUNA']} · "
                        f"{selected_patch['REGION']}"
                    )


            with class_column:

                st.caption(
                    "CLASE"
                )

                st.write(
                    f"{class_icon} "
                    f"**{class_label}**"
                )


            st.divider()


            detail_1, detail_2, detail_3 = (
                st.columns(3)
            )


            with detail_1:

                st.caption(
                    "Temporada"
                )

                st.write(
                    f"**{season_label}**"
                )

                st.caption(
                    "Fecha de referencia"
                )

                st.write(
                    f"**{format_date(selected_patch['fecha_referencia'])}**"
                )


            with detail_2:

                st.caption(
                    "Fecha objetivo pre-evento"
                )

                st.write(
                    f"**{format_date(selected_patch['fecha_objetivo_pre_evento'])}**"
                )

                st.caption(
                    "Imagen Sentinel-2"
                )

                st.write(
                    f"**{format_date(selected_patch['fecha_imagen_seleccionada'])}**"
                )


            with detail_3:

                st.caption(
                    "Píxeles válidos"
                )

                st.write(
                    f"**{format_percent(selected_patch['pct_pixeles_validos_patch'])}**"
                )

                st.caption(
                    "Grupo"
                )

                st.write(
                    f"**{selected_patch['group_id']}**"
                )


            group_size = int(
                selected_patch[
                    "n_patches_group_available"
                ]
            )

            if group_size > 1:

                st.info(
                    f"Este grupo contiene "
                    f"{group_size} parches disponibles."
                )