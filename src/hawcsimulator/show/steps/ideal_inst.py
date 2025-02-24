from __future__ import annotations

import xarray as xr
from showlib.l1b.data import L1bDataSet
from skretrieval.core.sasktranformat import SASKTRANRadiance

from hawcsimulator.datastructures.viewinggeo import ObservationContainer
from hawcsimulator.show.inst_model import L1bGeneratorILS


def l1b(
    calibration_database: xr.Dataset,
    observation: ObservationContainer,
    front_end_radiance: SASKTRANRadiance,
    l1b_cfg: dict | None = None,
) -> L1bDataSet:
    if l1b_cfg is None:
        l1b_cfg = {}

    l1b_gen = L1bGeneratorILS(
        calibration_database,
        observation.observation,
        **l1b_cfg,
    )
    return l1b_gen.run(front_end_radiance)
