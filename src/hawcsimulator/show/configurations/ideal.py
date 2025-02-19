from __future__ import annotations

import numpy as np
import xarray as xr
from showlib.l2.optical import h2o_optical_property
from showlib.processing.l1b_to_l2 import stratospheric_aerosol_optical_property

import hawcsimulator.show.steps.fer as fer
import hawcsimulator.show.steps.ideal_inst as ideal_inst
import hawcsimulator.show.steps.l2 as l2
import hawcsimulator.show.steps.por as por
from hawcsimulator.show.calibration import calibration_database
from hawcsimulator.simulator import Simulator


class IdealSHOWSimulator(Simulator):
    def __init__(self) -> None:
        super().__init__()

        self._modules.append(fer)
        self._modules.append(ideal_inst)
        self._modules.append(por)
        self._modules.append(l2)

    def _initialize_data(self) -> dict:
        data = {}
        data["calibration_database"] = xr.open_dataset(
            calibration_database("ideal", "v1")
        )

        data["viewing_tangent_altitudes"] = np.arange(0, 40001, 500.0)
        data["observer_altitude"] = 450000.0

        data["altitude_grid"] = np.arange(0, 65001, 1000.0)

        data["sample_wavelengths"] = (
            1e7 / data["calibration_database"]["sample_wavenumber"].to_numpy()
        )

        data["h2o_optical_property"] = h2o_optical_property()
        data["aerosol_optical_property"] = stratospheric_aerosol_optical_property()
        data["aerosol_kwargs"] = {"extinction_wavelength_nm": 1350.0}

        return data
