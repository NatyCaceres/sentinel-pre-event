from __future__ import annotations

import pandas as pd


class ExplorerSelectionError(Exception):
    """
    Error asociado a la selección de parches
    y grupos dentro del explorador.
    """


class PatchNotFoundError(ExplorerSelectionError):
    """
    El patch solicitado no existe en el catálogo.
    """


class GroupNotFoundError(ExplorerSelectionError):
    """
    El grupo solicitado no existe en el catálogo.
    """


def get_patch(
    catalog: pd.DataFrame,
    patch_id: str,
) -> pd.Series:
    """
    Obtiene un patch único desde el catálogo.
    """

    matches = catalog[
        catalog["patch_id"].astype(str)
        == str(patch_id)
    ]

    if matches.empty:
        raise PatchNotFoundError(
            f"No existe el patch '{patch_id}'."
        )

    if len(matches) > 1:
        raise ExplorerSelectionError(
            f"El patch_id '{patch_id}' "
            "no es único en el catálogo."
        )

    return matches.iloc[0].copy()


def get_group(
    catalog: pd.DataFrame,
    group_id: str,
) -> pd.DataFrame:
    """
    Obtiene todos los patches disponibles
    pertenecientes a un grupo.
    """

    matches = catalog[
        catalog["group_id"].astype(str)
        == str(group_id)
    ].copy()

    if matches.empty:
        raise GroupNotFoundError(
            f"No existe el grupo '{group_id}'."
        )

    return (
        matches
        .sort_values("patch_id")
        .reset_index(drop=True)
    )


def get_available_patch_ids(
    catalog: pd.DataFrame,
) -> list[str]:
    """
    Devuelve los patch_id disponibles ordenados.
    """

    return sorted(
        catalog["patch_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


def get_available_group_ids(
    catalog: pd.DataFrame,
) -> list[str]:
    """
    Devuelve los group_id disponibles ordenados.
    """

    return sorted(
        catalog["group_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


def get_group_patch_ids(
    catalog: pd.DataFrame,
    group_id: str,
) -> list[str]:
    """
    Devuelve los patch_id pertenecientes
    a un grupo determinado.
    """

    group = get_group(
        catalog,
        group_id,
    )

    return (
        group["patch_id"]
        .astype(str)
        .tolist()
    )