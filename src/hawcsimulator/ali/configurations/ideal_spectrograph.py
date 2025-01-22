from __future__ import annotations

import numpy as np
import xarray as xr

from hawcsimulator.ali import steps
from hawcsimulator.ali.calibration import calibration_database
from hawcsimulator.simulator import Simulator


class IdealALISimulator(Simulator):
    def __init__(self) -> None:
        s = [
            steps.GenerateFER(),
            steps.IdealALIModelL1b(),
            steps.ALIPORFromAtmosphere(),
            steps.ALIL1bToL2(),
        ]

        super().__init__(s)

    def _initialize_data(self, data: dict) -> dict:
        data["calibration_database"] = xr.open_dataset(
            calibration_database("ideal_spectrograph", "v1")
        ).load()

        data["calibration_database"].close()

        data["viewing_tangent_altitudes"] = np.arange(10000, 40001, 500.0)
        data["observer_altitude"] = 450000

        return data
