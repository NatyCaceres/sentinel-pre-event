from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.metadata_validator import (
    validate_phase1_file,
)
from src.data.phase1_manifest import (
    calculate_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PHASE1_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "phase1"
    / "phase1_clean.csv"
)

DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "phase1"
    / "phase1_manifest.json"
)


class Phase1DataError(Exception):
    """
    Error general asociado al dataset de Fase 1.
    """


class PatchNotFoundError(Phase1DataError):
    """
    Se genera cuando un patch_id no existe.
    """


class Phase1Dataset:
    """
    Interfaz de acceso al dataset limpio de Fase 1.

    La clase centraliza:

    - validación;
    - verificación de integridad;
    - carga de datos;
    - consultas sobre patches;
    - estadísticas generales.

    De esta forma, otras partes del producto no necesitan
    acceder directamente al CSV.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        dataset_path: Path,
        manifest: dict[str, Any],
    ) -> None:
        self._df = dataframe
        self._dataset_path = dataset_path
        self._manifest = manifest

    @classmethod
    def load(
        cls,
        dataset_path: Path | str = DEFAULT_PHASE1_PATH,
        manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
        verify_integrity: bool = True,
    ) -> "Phase1Dataset":
        """
        Carga un dataset de Fase 1 después de validarlo.

        Parameters
        ----------
        dataset_path:
            Ruta al CSV limpio de Fase 1.

        manifest_path:
            Ruta al manifest asociado al dataset.

        verify_integrity:
            Si es True, compara el SHA-256 actual del CSV
            con el registrado en el manifest.

        Returns
        -------
        Phase1Dataset
            Dataset validado y listo para consultas.
        """

        dataset_path = Path(dataset_path)
        manifest_path = Path(manifest_path)

        report = validate_phase1_file(dataset_path)

        if not report.is_valid:
            error_text = "; ".join(report.errors)

            raise Phase1DataError(
                "La metadata de Fase 1 no superó la validación. "
                f"Errores: {error_text}"
            )

        manifest = cls._load_manifest(manifest_path)

        if verify_integrity:
            cls._verify_integrity(
                dataset_path=dataset_path,
                manifest=manifest,
            )

        dataframe = pd.read_csv(dataset_path)

        return cls(
            dataframe=dataframe,
            dataset_path=dataset_path,
            manifest=manifest,
        )

    @staticmethod
    def _load_manifest(
        manifest_path: Path,
    ) -> dict[str, Any]:
        """
        Carga el manifest asociado al dataset.
        """

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No se encontró el manifest: {manifest_path}"
            )

        with manifest_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    @staticmethod
    def _verify_integrity(
        dataset_path: Path,
        manifest: dict[str, Any],
    ) -> None:
        """
        Comprueba que el archivo actual sea exactamente
        el mismo que fue registrado en el manifest.
        """

        expected_hash = manifest.get("sha256")

        if not expected_hash:
            raise Phase1DataError(
                "El manifest no contiene una huella SHA-256."
            )

        current_hash = calculate_sha256(dataset_path)

        if current_hash != expected_hash:
            raise Phase1DataError(
                "La integridad del dataset no pudo verificarse. "
                "El SHA-256 actual no coincide con el registrado "
                "en el manifest."
            )

    @property
    def dataset_id(self) -> str:
        """
        Identificador reproducible de la versión del dataset.
        """

        return self._manifest["dataset_id"]

    @property
    def dataframe(self) -> pd.DataFrame:
        """
        Devuelve una copia del DataFrame.

        Se retorna una copia para evitar modificaciones
        accidentales del dataset cargado internamente.
        """

        return self._df.copy()

    def summary(self) -> dict[str, Any]:
        """
        Devuelve un resumen general del dataset cargado.
        """

        group_sizes = self._df.groupby("group_id").size()

        seasons = (
            pd.to_numeric(
                self._df["season_year"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
        )

        return {
            "dataset_id": self.dataset_id,
            "rows": int(len(self._df)),
            "unique_patches": int(
                self._df["patch_id"].nunique()
            ),
            "unique_groups": int(
                self._df["group_id"].nunique()
            ),
            "multipatch_groups": int(
                (group_sizes > 1).sum()
            ),
            "class_distribution": (
                self._df["etiqueta"]
                .value_counts()
                .sort_index()
                .to_dict()
            ),
            "split_distribution": (
                self._df["split"]
                .value_counts()
                .to_dict()
            ),
            "season_min": int(seasons.min()),
            "season_max": int(seasons.max()),
        }

    def get_patch(
        self,
        patch_id: str,
    ) -> dict[str, Any]:
        """
        Obtiene la metadata correspondiente a un patch_id.

        Raises
        ------
        PatchNotFoundError
            Si el identificador no existe.
        """

        matches = self._df[
            self._df["patch_id"] == patch_id
        ]

        if matches.empty:
            raise PatchNotFoundError(
                f"No existe el patch_id '{patch_id}'."
            )

        return matches.iloc[0].to_dict()

    def get_group_patches(
        self,
        group_id: str,
    ) -> pd.DataFrame:
        """
        Obtiene todos los parches asociados a un group_id.

        Un grupo puede contener un único parche o varios
        cuando corresponde a un evento multiparche.

        Parameters
        ----------
        group_id:
            Identificador del grupo o evento.

        Returns
        -------
        pd.DataFrame
            Copia de los registros asociados al grupo.

        Raises
        ------
        Phase1DataError
            Si el group_id no existe.
        """

        matches = self._df[
            self._df["group_id"] == group_id
        ]

        if matches.empty:
            raise Phase1DataError(
                f"No existe el group_id '{group_id}'."
            )

        return matches.copy()
    def get_available_filters(self) -> dict[str, Any]:
        """
        Obtiene los valores disponibles para construir
        filtros dinámicos en la interfaz.

        Los valores se derivan directamente del dataset
        cargado para evitar opciones hardcodeadas.
        """

        seasons = (
            pd.to_numeric(
                self._df["season_year"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .sort_values()
            .unique()
            .tolist()
        )

        latitude_bands = (
            self._df["banda_lat"]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )

        splits = (
            self._df["split"]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )

        reference_dates = pd.to_datetime(
            self._df["fecha_referencia"],
            errors="coerce",
        ).dropna()

        if reference_dates.empty:
            reference_date_min = None
            reference_date_max = None
        else:
            reference_date_min = (
                reference_dates.min().date().isoformat()
            )
            reference_date_max = (
                reference_dates.max().date().isoformat()
            )

        return {
            "seasons": seasons,
            "labels": [0, 1],
            "latitude_bands": latitude_bands,
            "splits": splits,
            "reference_date_min": reference_date_min,
            "reference_date_max": reference_date_max,
        }
    def filter(
        self,
        season_year: int | None = None,
        label: int | None = None,
        latitude_band: str | None = None,
        split: str | None = None,
        reference_date_start: str | None = None,
        reference_date_end: str | None = None,
        multipatch_only: bool = False,
    ) -> pd.DataFrame:
        """
        Filtra las muestras de Fase 1 utilizando uno o más
        criterios combinables.

        Todos los filtros son opcionales. Cuando un parámetro
        es None, ese criterio no se aplica.

        Parameters
        ----------
        season_year:
            Año identificador de la temporada.

        label:
            Clase de la muestra:
            0 = negativa
            1 = positiva.

        latitude_band:
            Banda latitudinal del estudio.

        split:
            Partición del dataset: train o test.

        reference_date_start:
            Fecha mínima de referencia, inclusive,
            en formato YYYY-MM-DD.

        reference_date_end:
            Fecha máxima de referencia, inclusive,
            en formato YYYY-MM-DD.

        multipatch_only:
            Si es True, conserva únicamente muestras
            pertenecientes a grupos con más de un parche.

        Returns
        -------
        pd.DataFrame
            Copia del subconjunto filtrado.
        """

        filtered = self._df.copy()

        if season_year is not None:
            seasons = pd.to_numeric(
                filtered["season_year"],
                errors="coerce",
            )

            filtered = filtered[
                seasons == int(season_year)
            ]

        if label is not None:
            if label not in {0, 1}:
                raise Phase1DataError(
                    "La etiqueta debe ser 0 o 1."
                )

            labels = pd.to_numeric(
                filtered["etiqueta"],
                errors="coerce",
            )

            filtered = filtered[
                labels == label
            ]

        if latitude_band is not None:
            available_bands = set(
                self._df["banda_lat"]
                .dropna()
                .astype(str)
                .unique()
            )

            if latitude_band not in available_bands:
                raise Phase1DataError(
                    "Banda latitudinal no válida: "
                    f"'{latitude_band}'."
                )

            filtered = filtered[
                filtered["banda_lat"].astype(str)
                == latitude_band
            ]

        if split is not None:
            available_splits = set(
                self._df["split"]
                .dropna()
                .astype(str)
                .unique()
            )

            if split not in available_splits:
                raise Phase1DataError(
                    f"Split no válido: '{split}'."
                )

            filtered = filtered[
                filtered["split"].astype(str)
                == split
            ]

        reference_dates = pd.to_datetime(
            filtered["fecha_referencia"],
            errors="coerce",
        )

        if reference_date_start is not None:
            start_date = pd.to_datetime(
                reference_date_start,
                errors="coerce",
            )

            if pd.isna(start_date):
                raise Phase1DataError(
                    "La fecha inicial no es válida. "
                    "Utilice el formato YYYY-MM-DD."
                )

            filtered = filtered[
                reference_dates >= start_date
            ]

            reference_dates = reference_dates.loc[
                filtered.index
            ]

        if reference_date_end is not None:
            end_date = pd.to_datetime(
                reference_date_end,
                errors="coerce",
            )

            if pd.isna(end_date):
                raise Phase1DataError(
                    "La fecha final no es válida. "
                    "Utilice el formato YYYY-MM-DD."
                )

            filtered = filtered[
                reference_dates <= end_date
            ]

        if multipatch_only:
            group_sizes = (
                self._df
                .groupby("group_id")
                .size()
            )

            multipatch_groups = set(
                group_sizes[
                    group_sizes > 1
                ].index
            )

            filtered = filtered[
                filtered["group_id"].isin(
                    multipatch_groups
                )
            ]

        return (
            filtered
            .sort_values(
                by=[
                    "fecha_referencia",
                    "group_id",
                    "patch_id",
                ]
            )
            .reset_index(drop=True)
            .copy()
        )