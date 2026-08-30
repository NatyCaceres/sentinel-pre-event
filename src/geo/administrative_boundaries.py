from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_COMMUNES_PATH = (
    PROJECT_ROOT
    / "data"
    / "boundaries"
    / "chile_dpa_2023"
    / "raw"
    / "DPA_2023"
    / "COMUNAS"
    / "COMUNAS_v1.shp"
)

POINTS_CRS = "EPSG:4326"


class AdministrativeBoundaryError(Exception):
    """
    Error asociado al enriquecimiento territorial.
    """


def load_communes(
    path: Path | str = DEFAULT_COMMUNES_PATH,
) -> gpd.GeoDataFrame:
    """
    Carga la capa oficial de comunas utilizada
    para enriquecer geográficamente las muestras.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró la capa de comunas: {path}"
        )

    communes = gpd.read_file(path)

    required_columns = {
        "CUT_REG",
        "CUT_PROV",
        "CUT_COM",
        "REGION",
        "PROVINCIA",
        "COMUNA",
        "geometry",
    }

    missing_columns = (
        required_columns - set(communes.columns)
    )

    if missing_columns:
        raise AdministrativeBoundaryError(
            "La capa de comunas no cumple la estructura "
            "esperada. Faltan columnas: "
            f"{sorted(missing_columns)}"
        )

    if communes.crs is None:
        raise AdministrativeBoundaryError(
            "La capa de comunas no posee CRS definido."
        )

    return communes


def build_geographic_points(
    dataframe: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """
    Convierte latitud y longitud de Fase 1
    en geometrías de puntos.
    """

    required_columns = {
        "lat_centro",
        "lon_centro",
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise AdministrativeBoundaryError(
            "No se pueden construir puntos. "
            f"Faltan columnas: {sorted(missing_columns)}"
        )

    points = dataframe.copy()

    geometry = gpd.points_from_xy(
        points["lon_centro"],
        points["lat_centro"],
    )

    return gpd.GeoDataFrame(
        points,
        geometry=geometry,
        crs=POINTS_CRS,
    )


def enrich_with_administrative_boundaries(
    dataframe: pd.DataFrame,
    communes_path: Path | str = DEFAULT_COMMUNES_PATH,
) -> pd.DataFrame:
    """
    Asigna comuna, provincia y región a las muestras
    mediante una intersección espacial con la DPA 2023.

    Los registros que no intersectan una comuna
    permanecen en el resultado y son marcados como
    'unresolved'. No se realizan asignaciones por cercanía.
    """

    communes = load_communes(communes_path)

    points = build_geographic_points(dataframe)

    points = points.to_crs(communes.crs)

    administrative_columns = [
        "CUT_REG",
        "CUT_PROV",
        "CUT_COM",
        "REGION",
        "PROVINCIA",
        "COMUNA",
        "geometry",
    ]

    enriched = gpd.sjoin(
        points,
        communes[administrative_columns],
        how="left",
        predicate="within",
    )

    # Evitamos conservar columnas internas generadas
    # por el spatial join.
    enriched = enriched.drop(
        columns=["index_right"],
        errors="ignore",
    )

    enriched["administrative_status"] = (
        enriched["COMUNA"]
        .notna()
        .map(
            {
                True: "assigned",
                False: "unresolved",
            }
        )
    )

    # Volvemos a DataFrame normal porque la geometría
    # no es necesaria para almacenar la metadata derivada.
    enriched = pd.DataFrame(
        enriched.drop(columns="geometry")
    )

    return enriched.reset_index(drop=True)