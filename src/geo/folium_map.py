from __future__ import annotations

import folium
import pandas as pd
from folium.plugins import MarkerCluster


# ============================================================
# PALETA DEL MAPA
# ============================================================

POSITIVE_COLOR = "#D95D39"
NEGATIVE_COLOR = "#2563EB"
MIXED_CLUSTER_COLOR = "#344054"


class Phase1MapError(Exception):
    """
    Error asociado a la construcción del mapa de Fase 1.
    """


def extract_patch_id_from_popup(
    popup: object,
) -> str | None:
    """
    Extrae el patch_id del popup devuelto por st_folium.

    Streamlit-Folium entrega el contenido como texto plano. El
    contrato del popup sitúa la etiqueta ``PARCHE`` y el patch_id
    en líneas consecutivas.
    """

    if not isinstance(popup, str):
        return None

    lines = [
        line.strip()
        for line in popup.splitlines()
        if line.strip()
    ]

    for index, line in enumerate(lines[:-1]):
        if line == "PARCHE":
            return lines[index + 1]

    return None


def get_cluster_color(class_filter: str) -> str:
    """
    Define el color de los clusters según la clase visible.

    Cuando ambas clases están presentes se utiliza un color
    neutro para no sugerir incorrectamente que el cluster
    representa una sola clase.
    """

    if class_filter == "Con detección posterior":
        return POSITIVE_COLOR

    if class_filter == "Sin detección cercana":
        return NEGATIVE_COLOR

    return MIXED_CLUSTER_COLOR


def get_marker_color(label: int) -> str:
    """
    Devuelve el color correspondiente a una muestra individual.
    """

    if label == 1:
        return POSITIVE_COLOR

    if label == 0:
        return NEGATIVE_COLOR

    raise Phase1MapError(
        f"Etiqueta no reconocida: {label}"
    )


def format_reference_date(value) -> str:
    """
    Formatea la fecha de referencia para mostrarla
    de manera legible en el popup.
    """

    if pd.isna(value):
        return "Sin información"

    timestamp = pd.Timestamp(value)

    return timestamp.strftime("%d-%m-%Y")


def build_patch_popup(row) -> folium.Popup:
    """
    Construye el popup descriptivo de una muestra.
    """

    reference_date = format_reference_date(
        row.fecha_referencia
    )

    popup_html = f"""
    <div
        style="
            width:260px;
            font-family:Arial, sans-serif;
            color:#344054;
        "
    >
        <div
            style="
                font-size:11px;
                color:#98A2B3;
                font-weight:700;
                letter-spacing:0.04em;
            "
        >
            PARCHE
        </div>

        <div
            style="
                font-size:15px;
                font-weight:700;
                margin-bottom:12px;
            "
        >
            {row.patch_id}
        </div>

        <hr
            style="
                border:none;
                border-top:1px solid #EAECF0;
                margin-bottom:12px;
            "
        >

        <b>Comuna:</b> {row.COMUNA}<br>
        <b>Provincia:</b> {row.PROVINCIA}<br>
        <b>Región:</b> {row.REGION}<br><br>

        <b>Temporada:</b> {row.temporada}<br>
        <b>Clase:</b> {row.clase}<br>
        <b>Fecha de referencia:</b> {reference_date}<br><br>

        <span style="color:#667085;">
            Grupo: {row.group_id}
        </span>
    </div>
    """

    return folium.Popup(
        popup_html,
        max_width=310,
    )


def build_cluster_icon_function() -> str:
    """
    JavaScript utilizado por Leaflet para crear
    el icono visual de cada cluster.
    """

    return """
    function(cluster) {

        var count = cluster.getChildCount();

        return L.divIcon({
            html:
                '<div class="custom-cluster">' +
                    '<span>' + count + '</span>' +
                '</div>',
            className: '',
            iconSize: new L.Point(42, 42)
        });
    }
    """


def build_cluster_css(cluster_color: str) -> str:
    """
    Genera el estilo visual de los clusters.
    """

    return f"""
    <style>

    .custom-cluster {{
        width: 42px;
        height: 42px;
        border-radius: 50%;
        background: {cluster_color}33;
        display: flex;
        justify-content: center;
        align-items: center;
    }}

    .custom-cluster span {{
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: {cluster_color};
        color: white;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 12px;
        font-weight: 700;
        box-shadow: 0 2px 5px rgba(0,0,0,0.20);
    }}

    </style>
    """


def build_phase1_folium_map(
    points: pd.DataFrame,
    class_filter: str = "Todas",
) -> folium.Map:
    """
    Construye el mapa interactivo principal de Fase 1.

    Incluye:
    - OpenStreetMap;
    - agrupación de muestras mediante clusters;
    - color de cluster coherente con el filtro de clase;
    - área del cluster al pasar el cursor;
    - puntos individuales coloreados por clase;
    - popup con contexto territorial y metodológico.
    """

    if points.empty:
        raise Phase1MapError(
            "No existen puntos para representar en el mapa."
        )

    required_columns = {
        "patch_id",
        "group_id",
        "lat",
        "lon",
        "etiqueta",
        "clase",
        "temporada",
        "fecha_referencia",
        "COMUNA",
        "PROVINCIA",
        "REGION",
    }

    missing_columns = (
        required_columns - set(points.columns)
    )

    if missing_columns:
        raise Phase1MapError(
            "No se puede construir el mapa. "
            f"Faltan columnas: {sorted(missing_columns)}"
        )

    map_center = [
        float(points["lat"].mean()),
        float(points["lon"].mean()),
    ]

    map_object = folium.Map(
        location=map_center,
        zoom_start=5,
        tiles="OpenStreetMap",
        control_scale=True,
        prefer_canvas=True,
    )

    cluster_color = get_cluster_color(
        class_filter
    )

    cluster = MarkerCluster(
        name="Muestras",
        icon_create_function=(
            build_cluster_icon_function()
        ),

        # Conservamos esta funcionalidad porque permite
        # visualizar el alcance espacial del cluster.
        showCoverageOnHover=True,

        # Al hacer clic, Leaflet acerca automáticamente
        # la vista hacia las muestras del cluster.
        zoomToBoundsOnClick=True,

        # Cuando ya no puede aumentar más el zoom,
        # separa visualmente puntos coincidentes.
        spiderfyOnMaxZoom=True,
    ).add_to(map_object)

    cluster_css = build_cluster_css(
        cluster_color
    )

    map_object.get_root().html.add_child(
        folium.Element(cluster_css)
    )

    for row in points.itertuples():

        marker_color = get_marker_color(
            row.etiqueta
        )

        tooltip = (
            f"{row.COMUNA} · {row.patch_id}"
        )

        folium.CircleMarker(
            location=[
                row.lat,
                row.lon,
            ],
            radius=5,
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.85,
            weight=1.5,
            tooltip=tooltip,
            popup=build_patch_popup(row),
        ).add_to(cluster)

    # Ajuste automático a la extensión de los datos visibles.
    map_object.fit_bounds(
        [
            [
                float(points["lat"].min()),
                float(points["lon"].min()),
            ],
            [
                float(points["lat"].max()),
                float(points["lon"].max()),
            ],
        ],
        padding=(20, 20),
    )

    return map_object
