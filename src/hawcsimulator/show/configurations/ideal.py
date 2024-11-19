from __future__ import annotations

import numpy as np
import xarray as xr

from hawcsimulator.show import steps
from hawcsimulator.show.calibration import calibration_database
from hawcsimulator.simulator import Simulator


class IdealSHOWSimulator(Simulator):
    def __init__(self) -> None:
        s = [
            steps.GenerateFER(),
            steps.IdealSHOWModelL1b(),
            steps.SHOWPORFromAtmosphere(),
            steps.SHOWLibL1bToL2(),
        ]

        super().__init__(s)

    def _initialize_data(self, data: dict) -> dict:
        data["calibration_database"] = xr.open_dataset(
            calibration_database("ideal", "v1")
        )

        data["viewing_tangent_altitudes"] = np.arange(0, 40001, 500.0)
        data["observer_altitude"] = 450000

        data["sample_wavelengths"] = (
            1e7 / data["calibration_database"]["sample_wavenumber"].to_numpy()
        )

        return data
