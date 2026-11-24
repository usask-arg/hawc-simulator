from __future__ import annotations

import contextlib
import os
from pathlib import Path

import numpy as np
import xarray as xr
from showlib.cal_db import CalibrationDatabase
from showlib.l1b.data import L1bDataSet
from showlib.l2.config import load_retrieval_config
from showlib.l2.data import L2DataSet
from showlib.l2.lineshape import SHOWLineShapes, apodization_from_attrs
from showlib.processing.l1b_to_l2 import retrieve_l2
from threadpoolctl import threadpool_limits

from hawcsimulator.definitions import merge
from hawcsimulator.steps.truth import on_l2_grid


def default_line_shape_cache() -> Path:
    """$XDG_CACHE_HOME/hawcsimulator/show_line_shapes, or under ~/.cache."""
    root = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(root) / "hawcsimulator" / "show_line_shapes"


def cached_line_shapes(
    cal_db: CalibrationDatabase,
    apodization: float | None,
    model_res_cminv: float,
    cache_dir: str | os.PathLike,
) -> SHOWLineShapes:
    """
    The L2 line shapes of a calibration database, loaded from ``cache_dir`` or built
    and saved there. They are keyed by the calibration database's contents, the
    apodization and the model resolution, so a run with the same instrument loads
    them instead of building them (about 0.5 s per row).
    """
    apodization_key = "rect" if apodization is None else f"kaiser{apodization:g}"
    path = Path(cache_dir) / (
        f"{cal_db.fingerprint()[:24]}-{apodization_key}-res{model_res_cminv:g}"
    )
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # Another process may save them first
        with contextlib.suppress(FileExistsError):
            SHOWLineShapes(cal_db, apodization, model_res_cminv).save(path)
    return SHOWLineShapes.load(path, cal_db)


def l2(
    l1b: L1bDataSet,
    program_of_record: xr.Dataset,
    calibration_database: CalibrationDatabase,
    processing_definition: dict,
    provenance: dict,
    l2_cfg: dict | None = None,
    line_shape_cache: str | os.PathLike | None = None,
) -> L2DataSet:
    """
    The SHOW H2O retrieval of the L1B image, with the processing version's settings.
    The radiative transfer uses every CPU core unless the processing version or
    ``l2_cfg`` sets ``engine.num_threads``.

    Parameters
    ----------
    l2_cfg : dict | None, optional
        Overrides of the processing version's retrieval configuration, itself a set of
        overrides of showlib's (showlib/l2/retrieval_config.yaml, loaded with
        showlib.l2.config.load_retrieval_config), e.g. ``{"engine": {"num_threads": 4}}``.
        By default None
    line_shape_cache : str | os.PathLike | None, optional
        Directory the L2 line shapes are saved in and loaded from (see
        cached_line_shapes); None builds them for this run only. The SHOW simulator
        defaults it to default_line_shape_cache(). By default None

    Returns
    -------
    L2DataSet
        The L2 granule in showlib's L2 format, checked against its schema: along
        ``time`` and ``altitude``, h2o_vmr with its noise and solution errors, prior
        and averaging kernel, and the retrieved aerosol
    """
    defaults = {"engine": {"num_threads": os.cpu_count() or 1}}
    config = load_retrieval_config(
        merge(merge(defaults, processing_definition["l2"]), l2_cfg)
    )

    line_shapes = None
    if line_shape_cache is not None:
        line_shapes = cached_line_shapes(
            calibration_database,
            apodization_from_attrs(l1b.ds.attrs),
            config["model_res_cminv"],
            line_shape_cache,
        )
    # sasktran2 threads the radiative transfer; a multi-threaded BLAS competing
    # with it makes the retrieval slower
    with threadpool_limits(limits=1, user_api="blas"):
        profiles = retrieve_l2(
            l1b,
            program_of_record,
            calibration_database,
            config,
            line_shapes=line_shapes,
        )
    result = L2DataSet.from_profiles(profiles)
    result.ds.attrs.update(provenance)
    return result


def l2_truth(truth: xr.Dataset, l2: L2DataSet) -> xr.Dataset:
    """
    The true state on the L2's grid with the L2's names, so ``l2.ds - l2_truth`` is
    the retrieval error: h2o_vmr, and aerosol_extinction when the truth's aerosol is
    given at the L2's aerosol wavelength.
    """
    ds = l2.ds
    return on_l2_grid(
        truth,
        {"h2o_vmr": "h2o_vmr", "aerosol_extinction": "aerosol_extinction"},
        ds["h2o_vmr"],
        float(ds.attrs.get("aerosol_extinction_wavelength", np.nan)),
    )
