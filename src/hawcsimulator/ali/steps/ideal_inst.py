from __future__ import annotations

import xarray as xr
from aliprocessing.l1b.data import L1bImage
from skretrieval.core.sasktranformat import SASKTRANRadiance

from hawcsimulator.ali.inst_model import L1bGeneratorIdeal
from hawcsimulator.datastructures.viewinggeo import ObservationContainer


def l1b(
    calibration_database: xr.Dataset,
    observation: ObservationContainer,
    front_end_radiance: SASKTRANRadiance,
    polarization_states: list,
    l1b_cfg: dict | None = None,
) -> L1bImage:
    if l1b_cfg is None:
        l1b_cfg = {}

    l1b_gen = L1bGeneratorIdeal(
        calibration_database,
        observation.observation,
        pol_states=polarization_states,
        **l1b_cfg,
    )
    return l1b_gen.run(front_end_radiance)
