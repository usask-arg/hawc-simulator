from __future__ import annotations

import numpy as np
import xarray as xr
from aliprocessing.l1b.data import L1bImage
from aliprocessing.l2.config import load_retrieval_config
from aliprocessing.processing.l1b_to_l2 import (
    l2_granule,
    l2_profile,
    process_l1b_to_l2_image,
)
from threadpoolctl import threadpool_limits

from hawcsimulator.definitions import merge
from hawcsimulator.steps.truth import on_l2_grid


def l2(
    l1b: xr.Dataset,
    program_of_record: xr.Dataset,
    calibration_database: xr.Dataset,
    processing_definition: dict,
    provenance: dict,
    l1b_image_cfg: dict | None = None,
    l2_cfg: dict | None = None,
) -> xr.Dataset:
    """
    The ALI aerosol retrieval of the chosen observation sets and across-track bins of
    the L1B, with the processing version's settings.

    Parameters
    ----------
    l1b_image_cfg : dict | None, optional
        Overrides of the processing version's keyword arguments for
        aliprocessing.l1b.data.L1bImage.from_l1b: which ``observation_set`` (default
        0) and ``across_track`` bins (default the middle one) to retrieve, each an
        index, a list of indices or "all"; and low_alt, high_alt and max_dolp_noise.
        By default None
    l2_cfg : dict | None, optional
        Overrides of the processing version's retrieval configuration, itself a set of
        overrides of aliprocessing's (aliprocessing/l2/retrieval_config.yaml), e.g.
        ``{"minimizer": {"verbose": 0}}``. By default None

    Returns
    -------
    xr.Dataset
        The L2 granule in aliprocessing's L2 format, checked against its schema:
        along ``observation_set``, ``across_track`` and ``altitude``
    """
    cfg = {"observation_set": 0, "across_track": l1b.sizes["across_track"] // 2}
    cfg.update(merge(processing_definition["l1b_image"], l1b_image_cfg))
    observation_sets = _indices(
        cfg.pop("observation_set"), l1b.sizes["observation_set"]
    )
    across_track = _indices(cfg.pop("across_track"), l1b.sizes["across_track"])
    config = load_retrieval_config(merge(processing_definition["l2"], l2_cfg))
    por = program_of_record.isel(time=0)

    # sasktran2 threads the radiative transfer; a multi-threaded BLAS competing
    # with it makes the retrieval slower
    with threadpool_limits(limits=1, user_api="blas"):
        profiles = [
            [
                l2_profile(
                    process_l1b_to_l2_image(
                        L1bImage.from_l1b(l1b, observation_set, bin_, **cfg),
                        por,
                        calibration_database,
                        config,
                    ),
                    config,
                )
                for bin_ in across_track
            ]
            for observation_set in observation_sets
        ]
    result = l2_granule(profiles, l1b["across_track_angle"].to_numpy()[across_track])
    result.attrs.update(provenance)
    return result


def _indices(selection: int | list[int] | str, size: int) -> list[int]:
    """Indices chosen by an index, a list of indices or "all"."""
    if isinstance(selection, str):
        if selection != "all":
            msg = f'Expected an index, a list of indices or "all", got {selection!r}'
            raise ValueError(msg)
        return list(range(size))
    return [int(i) for i in np.atleast_1d(selection)]


def l2_truth(truth: xr.Dataset, l2: xr.Dataset) -> xr.Dataset:
    """
    The true state on the L2's grid with the L2's names, so ``l2 - l2_truth`` is the
    retrieval error: aerosol_extinction (when the truth's aerosol is given at the
    L2's aerosol wavelength), aerosol_median_radius and surface_albedo.
    """
    return on_l2_grid(
        truth,
        {
            "aerosol_extinction": "aerosol_extinction",
            "aerosol_median_radius": "aerosol_median_radius",
            "surface_albedo": "surface_albedo",
        },
        l2["aerosol_extinction"],
        float(l2.attrs.get("aerosol_extinction_wavelength", np.nan)),
    )
