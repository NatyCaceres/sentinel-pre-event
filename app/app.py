from __future__ import annotations

from functools import partial

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
    select_final_test_catalog,
)
from src.geo.folium_map import (
    build_phase1_folium_map,
    extract_patch_id_from_popup,
)
from src.geo.phase1_map import (
    build_map_points,
    format_season_label,
)
from src.spectral.inference import (
    SpectralInferenceError,
    load_spectral_model,
    predict_spectral_patch,
)
from src.spectral.patch import (
    SpectralPatchError,
    load_spectral_patch,
)
from src.spectral.storage import (
    SpectralStorageError,
    resolve_patch_tif,
)
from src.spectral.visualization import (
    SpectralVisualizationError,
    prepare_rgb,
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


def activate_patch_from_widget(
    widget_key: str,
) -> None:
    """Sincroniza una selección manual con el parche activo."""

    patch_id = st.session_state.get(
        widget_key
    )

    if patch_id is None:
        return

    st.session_state["active_patch_id"] = str(
        patch_id
    )
    st.session_state.pop(
        "spectral_prediction",
        None,
    )


def reset_group_patch_selection() -> None:
    """Evita conservar un parche al cambiar de grupo."""

    st.session_state.pop(
        "selected_group_patch",
        None,
    )


def activate_patch_from_map(
    map_key: str,
    valid_patch_ids: set[str],
) -> None:
    """Activa el punto clicado antes de renderizar la ficha."""

    map_state = st.session_state.get(
        map_key,
        {},
    )

    if not isinstance(map_state, dict):
        return

    clicked_patch_id = extract_patch_id_from_popup(
        map_state.get(
            "last_object_clicked_popup"
        )
    )

    if clicked_patch_id not in valid_patch_ids:
        return

    st.session_state["patch_search_mode"] = "Parche"
    st.session_state["manual_patch_id"] = clicked_patch_id
    st.session_state["active_patch_id"] = clicked_patch_id
    st.session_state.pop(
        "spectral_prediction",
        None,
    )


# ============================================================
# CATÁLOGO
# ============================================================

@st.cache_data
def load_catalog():
    return load_product_catalog()


@st.cache_resource
def load_cached_spectral_model():
    return load_spectral_model()


catalog = load_catalog()
test_catalog = select_final_test_catalog(
    catalog
)


# Solo el mapa necesita correspondencia territorial resuelta.
mapped_catalog = test_catalog[
    test_catalog["map_eligible"]
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
            "01 · FILTRAR MUESTRAS"
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
            '🟢 Test final congelado<br>'
            f'{len(test_catalog)} parches · 8 canales'
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
        'Explora las muestras del test final Sentinel-2 pre-evento, '
        'revisa su contexto territorial y posteriormente '
        'analiza sus características espectrales '
        'mediante deep learning.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write("")


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


    map_column, analysis_column = st.columns(
        [2.05, 1.0],
        gap="large",
    )


    # ========================================================
    # MAPA PRINCIPAL
    # ========================================================

    with map_column:

        st.markdown(
            '<div class="section-kicker">'
            '02 · SELECCIONAR EN EL MAPA'
            '</div>',
            unsafe_allow_html=True,
        )

        st.subheader(
            "Distribución de las muestras"
        )

        st.caption(
            "Acércate a un cluster y pulsa un punto para "
            "enviar esa muestra al panel de análisis."
        )

        st.caption(
            f"{len(points):,} muestras · "
            f"{visible_groups:,} grupos · "
            f"{visible_positive:,} con detección posterior · "
            f"{visible_negative:,} sin detección cercana"
        )


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

        visible_patch_ids = set(
            points["patch_id"].astype(str)
        )

        st_folium(
            map_object,
            height=650,
            use_container_width=True,
            returned_objects=[
                "last_object_clicked_popup",
            ],
            key=map_key,
            on_change=partial(
                activate_patch_from_map,
                map_key,
                visible_patch_ids,
            )
        )

    # ========================================================
    # PANEL DE SELECCIÓN Y ANÁLISIS
    # ========================================================

    with analysis_column:

        st.markdown(
            '<div class="section-kicker">'
            '03 · ANALIZAR MUESTRA'
            '</div>',
            unsafe_allow_html=True,
        )

        st.subheader(
            "Muestra seleccionada"
        )

        st.caption(
            "Selecciona un punto en el mapa o utiliza "
            "la búsqueda manual."
        )


        # ----------------------------------------------------
        # CATÁLOGO DE LA CONSULTA
        # ----------------------------------------------------

        filtered_catalog = test_catalog.copy()


        # ----------------------------------------------------
        # MODO DE BÚSQUEDA
        # ----------------------------------------------------

        with st.expander(
            f"Búsqueda manual en test final ({len(test_catalog)})",
            expanded=False,
        ):

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

            if search_mode == "Parche":

                patch_options = (
                    get_available_patch_ids(
                        filtered_catalog
                    )
                )

                selected_patch_id = st.selectbox(
                    "Patch ID",
                    options=patch_options,
                    index=None,
                    placeholder="Selecciona un parche",
                    key="manual_patch_id",
                    on_change=activate_patch_from_widget,
                    args=("manual_patch_id",),
                )

            else:

                group_options = (
                    get_available_group_ids(
                        filtered_catalog
                    )
                )

                selected_group_id = st.selectbox(
                    "Grupo",
                    options=group_options,
                    index=None,
                    placeholder="Selecciona un grupo",
                    key="selected_group",
                    on_change=reset_group_patch_selection,
                )

                if selected_group_id is not None:

                    group_patch_options = (
                        get_group_patch_ids(
                            filtered_catalog,
                            selected_group_id,
                        )
                    )

                    selected_patch_id = st.selectbox(
                        "Parche del grupo",
                        options=group_patch_options,
                        index=None,
                        placeholder="Selecciona un parche",
                        key="selected_group_patch",
                        on_change=activate_patch_from_widget,
                        args=("selected_group_patch",),
                    )


        selected_patch_id = st.session_state.get(
            "active_patch_id"
        )

        if selected_patch_id is None:
            st.info(
                "Selecciona un punto del mapa o busca un "
                "parche del test final para comenzar."
            )
            st.stop()


        # ----------------------------------------------------
        # CARGAR PATCH
        # ----------------------------------------------------

        selected_patch = get_patch(
            test_catalog,
            selected_patch_id,
        )


        spectral_patch = None
        spectral_patch_error = None

        try:

            selected_tif_path = resolve_patch_tif(
                selected_patch
            )

            spectral_patch = load_spectral_patch(
                selected_tif_path
            )

        except (
            FileNotFoundError,
            SpectralStorageError,
            SpectralPatchError,
        ) as error:

            spectral_patch_error = error


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
            key=(
                "selected_patch_card_"
                f"{selected_patch_id}"
            ),
        ):

            title_column = st.container()
            class_column = st.container()

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
                    "CLASE OBSERVADA"
                )

                st.write(
                    f"{class_icon} "
                    f"**{class_label}**"
                )


            st.divider()


            detail_1 = st.container()
            detail_2 = st.container()
            detail_3 = st.container()


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


        # ----------------------------------------------------
        # VISTA RGB OPCIONAL DEL PATCH
        # ----------------------------------------------------

        st.write("")

        with st.expander(
            "Ver imagen RGB del parche seleccionado"
        ):

            if spectral_patch is None:

                st.error(
                    "No fue posible cargar la imagen "
                    "del parche seleccionado."
                )

                st.caption(
                    str(spectral_patch_error)
                )

            else:

                try:

                    rgb_image = prepare_rgb(
                        spectral_patch
                    )

                    st.image(
                        rgb_image,
                        caption=(
                            "RGB natural B4/B3/B2. "
                            "Contraste percentil 2–98 aplicado "
                            "exclusivamente para visualización."
                        ),
                        width=256,
                    )

                except SpectralVisualizationError as error:

                    st.error(
                        "No fue posible construir la vista RGB."
                    )

                    st.caption(
                        str(error)
                    )


        # ----------------------------------------------------
        # INFERENCIA DEL MODELO
        # ----------------------------------------------------

        st.write("")

        with st.container(
            border=True,
        ):

            st.caption(
                "ANÁLISIS MEDIANTE RESNET18"
            )

            st.subheader(
                "Clasificar el patrón espectral"
            )

            st.caption(
                "El modelo procesa los ocho canales del "
                "GeoTIFF seleccionado con el protocolo "
                "congelado de Fase 3."
            )

            analyze_patch = st.button(
                "Analizar patrón espectral",
                type="primary",
                key="analyze_spectral_patch",
                disabled=spectral_patch is None,
            )

            prediction_state_key = (
                "spectral_prediction"
            )

            if analyze_patch:

                st.session_state.pop(
                    prediction_state_key,
                    None,
                )

                try:

                    with st.spinner(
                        "Cargando y analizando el parche..."
                    ):

                        loaded_model = (
                            load_cached_spectral_model()
                        )

                        prediction = (
                            predict_spectral_patch(
                                spectral_patch,
                                loaded_model,
                            )
                        )

                    st.session_state[
                        prediction_state_key
                    ] = {
                        "patch_id": str(
                            selected_patch[
                                "patch_id"
                            ]
                        ),
                        "prediction": prediction,
                    }

                except (
                    FileNotFoundError,
                    SpectralStorageError,
                    SpectralPatchError,
                    SpectralInferenceError,
                ) as error:

                    st.error(
                        "No fue posible analizar el parche "
                        "seleccionado."
                    )

                    st.caption(
                        str(error)
                    )

            prediction_state = (
                st.session_state.get(
                    prediction_state_key
                )
            )

            current_patch_id = str(
                selected_patch["patch_id"]
            )

            if (
                prediction_state is not None
                and prediction_state["patch_id"]
                == current_patch_id
            ):

                prediction = prediction_state[
                    "prediction"
                ]

                score_0, score_1 = (
                    prediction.probabilities
                )

                score_column_0, score_column_1 = (
                    st.columns(2)
                )

                score_column_0.metric(
                    "Score clase 0",
                    f"{score_0:.1%}",
                )

                score_column_1.metric(
                    "Score clase 1",
                    f"{score_1:.1%}",
                )

                predicted_label = (
                    "Patrón asociado a clase 1"
                    if prediction.predicted_class == 1
                    else "Patrón asociado a clase 0"
                )

                st.info(
                    f"**Clasificación del modelo:** "
                    f"{predicted_label}."
                )

                st.caption(
                    "Decisión obtenida con el umbral "
                    f"congelado de validación "
                    f"({prediction.threshold:.3f})."
                )

                st.warning(
                    "Este resultado clasifica patrones "
                    "espectrales pre-evento según su "
                    "asociación con las clases del estudio. "
                    "No predice incendios ni representa "
                    "riesgo absoluto."
                )
