from __future__ import annotations

from html import escape
from textwrap import dedent

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.data.explorer_filters import (
    apply_explorer_filters,
    get_reference_date_bounds,
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
from src.ui.spectral_preview import render_spectral_preview
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


def show_explorer_view() -> None:
    """Navega a la selección cartográfica de muestras."""

    st.session_state["main_view"] = (
        "01 · Explorar mapa"
    )


def show_analysis_view() -> None:
    """Navega al análisis de la muestra activa."""

    st.session_state["main_view"] = (
        "02 · Analizar muestra"
    )


def commit_pending_map_patch(
) -> None:
    """Confirma una selección del mapa antes de renderizar."""

    patch_id = st.session_state.pop(
        "pending_map_patch_id",
        None,
    )

    if patch_id is None:
        return

    st.session_state["patch_search_mode"] = "Parche"
    st.session_state["manual_patch_id"] = str(patch_id)
    st.session_state["active_patch_id"] = str(patch_id)
    st.session_state.pop(
        "spectral_prediction",
        None,
    )


commit_pending_map_patch()


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


# Filtrar primero el test completo conserva los casos sin ubicación.
all_points = build_map_points(test_catalog)
all_points["map_eligible"] = all_points["patch_id"].isin(
    test_catalog.loc[test_catalog["map_eligible"], "patch_id"]
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

# ============================================================
# BARRA LATERAL DE EXPLORACIÓN
# ============================================================

with st.sidebar:

    st.caption(
        "CONFIGURAR EXPLORACIÓN"
    )

    st.header(
        "Seleccionar muestras"
    )

    st.caption(
        "Filtra por temporada, región o clase. También "
        "puedes buscar una muestra específica."
    )

    # Fuera del formulario: las fechas deben reaccionar a la temporada.
    selected_season_label = st.selectbox(
        "Temporada",
        options=["Todas"] + list(season_labels.keys()),
        index=0,
        key="query_season",
    )
    minimum_reference_date, maximum_reference_date = get_reference_date_bounds(
        all_points, season_labels.get(selected_season_label)
    )
    if st.session_state.get("date_bounds_season") != selected_season_label:
        st.session_state["query_dates"] = (
            minimum_reference_date, maximum_reference_date
        )
        st.session_state["date_bounds_season"] = selected_season_label

    with st.form(
        "explorer_query_form",
        border=False,
    ):

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
            st.caption(
                "Fechas disponibles en la temporada: "
                f"{minimum_reference_date:%d-%m-%Y} a "
                f"{maximum_reference_date:%d-%m-%Y}."
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

        st.form_submit_button(
            "Actualizar mapa",
            type="primary",
            use_container_width=True,
        )

    st.divider()

    st.caption(
        "BÚSQUEDA DIRECTA"
    )

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

        patch_options = get_available_patch_ids(
            test_catalog
        )

        st.selectbox(
            "ID del parche",
            options=patch_options,
            index=None,
            placeholder="Selecciona un parche",
            key="manual_patch_id",
            on_change=activate_patch_from_widget,
            args=("manual_patch_id",),
        )

    else:

        group_options = get_available_group_ids(
            test_catalog
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

            group_patch_options = get_group_patch_ids(
                test_catalog,
                selected_group_id,
            )

            st.selectbox(
                "Parche del grupo",
                options=group_patch_options,
                index=None,
                placeholder="Selecciona un parche",
                key="selected_group_patch",
                on_change=activate_patch_from_widget,
                args=("selected_group_patch",),
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

filtered_samples = apply_explorer_filters(
    all_points,
    season_year=selected_season_year,
    region=region_filter,
    label=class_value,
    reference_start=reference_start,
    reference_end=reference_end,
    multipatch_only=multipatch_only,
)
points = filtered_samples.loc[filtered_samples["map_eligible"]].copy()


# ============================================================
# CONTENIDO PRINCIPAL
# ============================================================

with st.container():

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
            '🟢 Análisis espectral del terreno<br>'
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

    st.caption("Prototipo de exploración y revisión experimental")

    st.markdown(
        '<div class="product-subtitle">'
        'Explora las muestras del conjunto de prueba, consulta sus antecedentes '
        'y reproduce la clasificación del modelo seleccionado.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write("")

    main_view = st.segmented_control(
        "Etapa del flujo",
        options=[
            "01 · Explorar mapa",
            "02 · Analizar muestra",
        ],
        default="01 · Explorar mapa",
        key="main_view",
        label_visibility="collapsed",
    )

    if main_view is None:
        main_view = "01 · Explorar mapa"

    selected_patch_id = st.session_state.get(
        "active_patch_id"
    )

    selected_patch = (
        get_patch(
            test_catalog,
            selected_patch_id,
        )
        if selected_patch_id is not None
        else None
    )


    # --------------------------------------------------------
    # VISTA DE ANÁLISIS
    # --------------------------------------------------------

    if main_view == "02 · Analizar muestra":

        st.markdown(
            '<div class="section-kicker">'
            '02 · ANALIZAR MUESTRA'
            '</div>',
            unsafe_allow_html=True,
        )

        heading_column, back_column = st.columns(
            [4, 1]
        )

        with heading_column:

            st.subheader("Muestra seleccionada")

            st.caption(
                "Revisa sus antecedentes y ejecuta el modelo "
                "sobre los ocho canales del GeoTIFF."
            )

        with back_column:

            st.button(
                "← Volver al mapa",
                on_click=show_explorer_view,
                use_container_width=True,
            )

        if selected_patch is None:

            st.info(
                "Primero selecciona un punto del mapa o busca "
                "un parche del test final en la barra lateral."
            )

            st.button(
                "Explorar muestras",
                type="primary",
                on_click=show_explorer_view,
            )

            st.stop()

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
            if int(selected_patch["etiqueta"]) == 1
            else "Sin detección cercana"
        )

        class_icon = (
            "🔴"
            if int(selected_patch["etiqueta"]) == 1
            else "🔵"
        )

        season_label = format_season_label(
            int(selected_patch["season_year"])
        )

        patch_label = escape(
            str(selected_patch["patch_id"])
        )
        safe_class_label = escape(class_label)
        location_label = (
            escape(str(selected_patch["COMUNA"]))
            if pd.notna(selected_patch["COMUNA"])
            else "Sin asignación administrativa"
        )
        region_label = (
            escape(str(selected_patch["REGION"]))
            if pd.notna(selected_patch["REGION"])
            else "Ubicación no disponible"
        )
        observed_class_modifier = (
            "positive"
            if int(selected_patch["etiqueta"]) == 1
            else "negative"
        )
        group_size = int(
            selected_patch["n_patches_group_available"]
        )
        group_note = (
            f"{group_size} parches disponibles en este grupo"
            if group_size > 1
            else "Muestra única disponible en este grupo"
        )

        st.markdown(
            dedent(
                f"""
            <section class="sample-profile">
                <div class="sample-profile__hero">
                    <div>
                        <div class="sample-profile__eyebrow">
                            PARCHE SELECCIONADO
                        </div>
                        <div class="sample-profile__id">{patch_label}</div>
                        <div class="sample-profile__place">
                            {location_label} · {region_label}
                        </div>
                    </div>
                    <div class="observed-class observed-class--{observed_class_modifier}">
                        <span class="observed-class__label">CLASE OBSERVADA</span>
                        <span class="observed-class__value">
                            {class_icon} {safe_class_label}
                        </span>
                    </div>
                </div>
                <div class="sample-profile__facts">
                    <div><span>Temporada</span><strong>{escape(season_label)}</strong></div>
                    <div><span>Fecha de referencia</span><strong>{escape(format_date(selected_patch['fecha_referencia']))}</strong></div>
                    <div><span>Fecha objetivo pre-evento</span><strong>{escape(format_date(selected_patch['fecha_objetivo_pre_evento']))}</strong></div>
                    <div><span>Fecha de adquisición Sentinel-2</span><strong>{escape(format_date(selected_patch['fecha_imagen_seleccionada']))}</strong></div>
                    <div><span>Píxeles válidos</span><strong>{escape(format_percent(selected_patch['pct_pixeles_validos_patch']))}</strong></div>
                    <div><span>Grupo</span><strong>{escape(str(selected_patch['group_id']))}</strong></div>
                </div>
                <div class="sample-profile__note">{escape(group_note)}</div>
            </section>
            """
            ),
            unsafe_allow_html=True,
        )

        # Conserva la posición visual; se llena después de procesar inferencia
        # para que el botón Grad-CAM se habilite en esa misma ejecución.
        spectral_preview = st.container(key="rgb_preview_panel")

        st.write("")

        with st.container(
            border=False,
            key="prediction_panel",
        ):

            st.markdown(
                dedent(
                    """
                <div class="prediction-intro">
                    <div class="prediction-intro__mark">IA</div>
                    <div>
                        <div class="prediction-intro__eyebrow">
                            ANÁLISIS MEDIANTE RESNET18
                        </div>
                        <div class="prediction-intro__title">
                            Clasificar el patrón espectral
                        </div>
                        <div class="prediction-intro__copy">
                            El modelo analiza los ocho canales del GeoTIFF
                            y fue validado durante la Fase 3.
                        </div>
                    </div>
                </div>
                """
                ),
                unsafe_allow_html=True,
            )

            analyze_patch = st.button(
                "Analizar patrón espectral",
                type="primary",
                key="analyze_spectral_patch",
                disabled=spectral_patch is None,
            )

            prediction_state_key = "spectral_prediction"

            if analyze_patch:

                st.session_state.pop("spectral_gradcam", None)
                st.session_state.pop(
                    prediction_state_key,
                    None,
                )

                try:

                    with st.spinner(
                        "Cargando y analizando el parche..."
                    ):

                        loaded_model = load_cached_spectral_model()
                        prediction = predict_spectral_patch(
                            spectral_patch,
                            loaded_model,
                        )

                    st.session_state[prediction_state_key] = {
                        "patch_id": str(selected_patch["patch_id"]),
                        "prediction": prediction,
                        "model_sha256": loaded_model.contract["model"]["checkpoint_sha256"],
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
                    st.caption(str(error))

            prediction_state = st.session_state.get(
                prediction_state_key
            )
            current_patch_id = str(selected_patch["patch_id"])

            if (
                prediction_state is not None
                and prediction_state["patch_id"]
                == current_patch_id
            ):

                prediction = prediction_state["prediction"]
                score_0, score_1 = prediction.probabilities

                predicted_label = (
                    "Patrón asociado a clase 1"
                    if prediction.predicted_class == 1
                    else "Patrón asociado a clase 0"
                )

                score_0_percent = max(
                    0.0,
                    min(100.0, float(score_0) * 100),
                )
                score_1_percent = max(
                    0.0,
                    min(100.0, float(score_1) * 100),
                )
                predicted_modifier = (
                    "positive"
                    if prediction.predicted_class == 1
                    else "negative"
                )

                st.markdown(
                    dedent(
                        f"""
                    <div class="prediction-result prediction-result--{predicted_modifier}">
                        <div class="prediction-result__eyebrow">
                            CLASIFICACIÓN DEL MODELO
                        </div>
                        <div class="prediction-result__title">
                            {escape(predicted_label)}
                        </div>
                        <p>Probabilidades de clase estimadas por el modelo</p>
                        <div class="probability-row">
                            <div class="probability-row__label">
                                <span>Clase 0 · Sin detección cercana</span>
                                <strong>{score_0_percent:.1f} %</strong>
                            </div>
                            <div class="probability-track">
                                <span class="probability-fill probability-fill--negative"
                                      style="width: {score_0_percent:.3f}%"></span>
                            </div>
                        </div>
                        <div class="probability-row">
                            <div class="probability-row__label">
                                <span>Clase 1 · Con detección posterior</span>
                                <strong>{score_1_percent:.1f} %</strong>
                            </div>
                            <div class="probability-track">
                                <span class="probability-fill probability-fill--positive"
                                      style="width: {score_1_percent:.3f}%"></span>
                            </div>
                        </div>
                    </div>
                    """
                    ),
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Estos porcentajes no representan probabilidad de "
                    "ocurrencia de incendio."
                )
                st.caption(
                    "Decisión obtenida con el umbral "
                    f"congelado de validación "
                    f"({prediction.threshold:.3f})."
                )
                st.markdown(
                    dedent(
                        """
                        <div class="scope-note">
                            <strong>Alcance del resultado</strong>
                            <span>
                                Clasifica patrones espectrales pre-evento
                                según su asociación con las clases del estudio.
                                No predice incendios ni representa riesgo absoluto.
                            </span>
                        </div>
                        """
                    ),
                    unsafe_allow_html=True,
                )

        with spectral_preview:
            render_spectral_preview(
                spectral_patch,
                current_patch_id,
                prediction_state,
                load_cached_spectral_model,
                error=spectral_patch_error,
            )

        st.stop()


    # --------------------------------------------------------
    # CONTEOS DE LA CONSULTA, INCLUIDOS LOS CASOS SIN UBICACIÓN
    # --------------------------------------------------------

    test_positive = int((filtered_samples["etiqueta"] == 1).sum())
    test_negative = int((filtered_samples["etiqueta"] == 0).sum())
    unmapped_test = filtered_samples.loc[~filtered_samples["map_eligible"]]

    if not unmapped_test.empty:
        unmapped_details = "; ".join(
            f"{row.patch_id} (clase {int(row.etiqueta)})"
            for row in unmapped_test.itertuples(index=False)
        )
    # ========================================================
    # MAPA PRINCIPAL
    # ========================================================

    with st.container():

        st.markdown(
            '<div class="section-kicker">'
            'EXPLORACIÓN GEOGRÁFICA'
            '</div>',
            unsafe_allow_html=True,
        )

        st.subheader(
            "Explorar muestras en el mapa"
        )

        st.caption(
            "Acércate a una agrupación de puntos y selecciona uno para "
            "revisar su muestra."
        )

        if class_filter == "Todas":
            legend_items = (
                '<span><i class="legend-dot legend-dot--positive"></i>'
                'Con detección posterior</span>'
                '<span><i class="legend-dot legend-dot--negative"></i>'
                'Sin detección cercana</span>'
                '<span><i class="legend-dot legend-dot--cluster"></i>'
                'Agrupación de puntos</span>'
            )
        elif class_filter == "Con detección posterior":
            legend_items = (
                '<span><i class="legend-dot legend-dot--positive"></i>'
                'Con detección posterior</span>'
                '<span><i class="legend-dot legend-dot--cluster"></i>'
                'Agrupación de puntos</span>'
            )
        else:
            legend_items = (
                '<span><i class="legend-dot legend-dot--negative"></i>'
                'Sin detección cercana</span>'
                '<span><i class="legend-dot legend-dot--cluster"></i>'
                'Agrupación de puntos</span>'
            )

        st.html(
            dedent(
                f"""
                <div class="map-context-bar">
                    <div class="map-context-bar__count">
                        <strong>Consulta actual</strong>
                        <span>·</span>
                        {test_positive:,} clase 1
                        <span>·</span>
                        {test_negative:,} clase 0
                    </div>
                    <div class="map-context-bar__legend">
                        {legend_items}
                    </div>
                </div>
                """
            ),
        )

        # ----------------------------------------------------
        # MAPA
        # ----------------------------------------------------

        if points.empty:
            st.warning(
                "No existen muestras cartografiables que cumplan "
                "la combinación de filtros seleccionada."
            )
            st.stop()

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

        map_state = st_folium(
            map_object,
            height=560,
            use_container_width=True,
            returned_objects=[
                "last_object_clicked_popup",
            ],
            key=map_key,
        )

        clicked_patch_id = extract_patch_id_from_popup(
            map_state.get(
                "last_object_clicked_popup"
            )
            if isinstance(map_state, dict)
            else None
        )

        map_selection_token = (
            map_key,
            clicked_patch_id,
        )

        if (
            clicked_patch_id in visible_patch_ids
            and map_selection_token
            != st.session_state.get(
                "last_processed_map_selection"
            )
        ):
            st.session_state[
                "pending_map_patch_id"
            ] = clicked_patch_id
            st.session_state[
                "last_processed_map_selection"
            ] = map_selection_token
            st.rerun()

    # ========================================================
    # RESUMEN DE LA MUESTRA ACTIVA
    # ========================================================

    st.write("")

    st.markdown(
        '<div class="section-kicker">'
        'MUESTRA SELECCIONADA'
        '</div>',
        unsafe_allow_html=True,
    )

    if selected_patch is None:

        st.info(
            "Selecciona un punto del mapa o busca un "
            "parche del test final para continuar."
        )

    else:

        selected_class_label = (
            "Con detección posterior"
            if int(selected_patch["etiqueta"]) == 1
            else "Sin detección cercana"
        )

        selected_class_icon = (
            "🔴"
            if int(selected_patch["etiqueta"]) == 1
            else "🔵"
        )
        summary_location = (
            escape(str(selected_patch["COMUNA"]))
            if pd.notna(selected_patch["COMUNA"])
            else "Sin asignación administrativa"
        )
        summary_region = (
            escape(str(selected_patch["REGION"]))
            if pd.notna(selected_patch["REGION"])
            else "Ubicación no disponible"
        )

        with st.container(
            border=False,
            key=(
                "explorer_selection_panel_"
                f"{selected_patch_id}"
            ),
        ):

            summary_column, action_column = (
                st.columns(
                    [4, 1.25],
                    gap="large",
                    vertical_alignment="center",
                )
            )

            with summary_column:

                st.markdown(
                    dedent(
                        f"""
                        <div class="active-sample">
                            <div class="active-sample__eyebrow">LISTA PARA ANALIZAR</div>
                            <div class="active-sample__content">
                                <div class="active-sample__id">
                                    {escape(str(selected_patch['patch_id']))}
                                </div>
                                <div class="active-sample__place">
                                    {summary_location} · {summary_region}
                                </div>
                            </div>
                        </div>
                        """
                    ),
                    unsafe_allow_html=True,
                )

            with action_column:

                st.button(
                    "Revisar y analizar →",
                    type="primary",
                    on_click=show_analysis_view,
                    use_container_width=True,
                )
