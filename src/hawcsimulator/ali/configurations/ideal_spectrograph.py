from __future__ import annotations

import numpy as np
import xarray as xr
from aliprocessing.l2.optical import aerosol_median_radius_db

import hawcsimulator.ali.steps.fer as fer
import hawcsimulator.ali.steps.ideal_inst as ideal_inst
import hawcsimulator.ali.steps.l2 as l2
import hawcsimulator.ali.steps.por as por
from hawcsimulator.ali.calibration import calibration_database
from hawcsimulator.simulator import Simulator


class IdealALISimulator(Simulator):
    def __init__(self) -> None:
        super().__init__()
        self._modules.append(fer)
        self._modules.append(ideal_inst)
        self._modules.append(por)
        self._modules.append(l2)

    def _initialize_data(self) -> dict:
        data = {}
        data["calibration_database"] = xr.open_dataset(
            calibration_database("ideal_spectrograph", "v1")
        ).load()

        data["calibration_database"].close()

        data["viewing_tangent_altitudes"] = np.arange(10000, 40001, 500.0)
        data["observer_altitude"] = 450000.0

        data["aerosol_optical_property"] = aerosol_median_radius_db()
        data["aerosol_kwargs"] = {"extinction_wavelength_nm": 745.0}

        return data
