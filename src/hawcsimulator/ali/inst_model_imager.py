from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from aliprocessing.l1b.data import L1bImage, L1bSpectra

from hawcsimulator.ali.inst_model import L1bGenerator


class L1bGeneratorIdealImager(L1bGenerator):
    def __init__(
        self,
        cal_db: xr.Dataset,  # noqa: ARG002
        observation,
        noise_model=None,
        pol_states=None,
        include_noise=False,
        **kwargs,
    ):
        self._observation = observation
        self._noise = include_noise
        self._noise_model = noise_model

        self._pol_angles = [-60.0, 0.0, 60.0]
        if pol_states is None:
            self._pol_states = ["I", "dolp"]
        else:
            self._pol_states = pol_states

    def run(self, fer: xr.Dataset):
        result = {}

        I = fer.data["radiance"].isel(stokes=0)  # noqa: E741
        Q = fer.data["radiance"].isel(stokes=1)
        U = fer.data["radiance"].isel(stokes=2)

        measurements = []
        errors = []
        mueller = []
        for a in self._pol_angles:
            mueller.append([1, np.cos(np.deg2rad(2 * a)), np.sin(np.deg2rad(2 * a))])
            m = (
                I + np.cos(np.deg2rad(2 * a)) * Q + np.sin(np.deg2rad(2 * a)) * U
            ) / 2.0

            m, sigma = self._noise_model.calc_noise(m)

            sigma = sigma**2

            errors.append(sigma)

            measurements.append(m)

        mueller = np.array(mueller) / 2.0

        # Add noise to measurements

        inv_mueller = np.linalg.inv(mueller)

        recovered_I = (
            measurements[0] * inv_mueller[0, 0]
            + measurements[1] * inv_mueller[0, 1]
            + measurements[2] * inv_mueller[0, 2]
        )
        recovered_Q = (
            measurements[0] * inv_mueller[1, 0]
            + measurements[1] * inv_mueller[1, 1]
            + measurements[2] * inv_mueller[1, 2]
        )
        recovered_U = (
            measurements[0] * inv_mueller[2, 0]
            + measurements[1] * inv_mueller[2, 1]
            + measurements[2] * inv_mueller[2, 2]
        )

        dolp = np.sqrt(recovered_Q**2 + recovered_U**2) / recovered_I

        # error prop for DOLP
        inv2 = inv_mueller**2
        var_I = inv2[0, 0] * errors[0] + inv2[0, 1] * errors[1] + inv2[0, 2] * errors[2]
        var_Q = inv2[1, 0] * errors[0] + inv2[1, 1] * errors[1] + inv2[1, 2] * errors[2]
        var_U = inv2[2, 0] * errors[0] + inv2[2, 1] * errors[1] + inv2[2, 2] * errors[2]

        den = np.sqrt(recovered_Q**2 + recovered_U**2)
        dQ = recovered_Q / (den * recovered_I)
        dU = recovered_U / (den * recovered_I)
        dI = -den / (recovered_I**2)

        var_dolp = (dQ**2) * var_Q + (dU**2) * var_U + (dI**2) * var_I

        result["I"] = L1bSpectra.from_np_arrays(
            recovered_I.to_numpy(),
            np.sqrt(var_I.to_numpy()),
            fer.data["tangent_altitude"].to_numpy(),
            fer.data["tangent_latitude"].to_numpy(),
            fer.data["tangent_longitude"].to_numpy(),
            fer.data["wavelength_nm"].to_numpy(),
            pd.to_datetime(fer.data["time"].to_numpy()[0]),
            0.0,
            0.0,
            float(fer.data["observer_altitude"].to_numpy()[0]),
            np.rad2deg(np.arccos(fer.data["tangent_cos_sza"].to_numpy())),
            fer.data["tangent_solar_azimuth"].to_numpy(),
            fer.data["tangent_observer_azimuth"].to_numpy(),
        )

        if "dolp" in self._pol_states:
            result["dolp"] = L1bSpectra.from_np_arrays(
                dolp.to_numpy(),
                np.sqrt(var_dolp.to_numpy()),
                fer.data["tangent_altitude"].to_numpy(),
                fer.data["tangent_latitude"].to_numpy(),
                fer.data["tangent_longitude"].to_numpy(),
                fer.data["wavelength_nm"].to_numpy(),
                pd.to_datetime(fer.data["time"].to_numpy()[0]),
                0.0,
                0.0,
                float(fer.data["observer_altitude"].to_numpy()[0]),
                np.rad2deg(np.arccos(fer.data["tangent_cos_sza"].to_numpy())),
                fer.data["tangent_solar_azimuth"].to_numpy(),
                fer.data["tangent_observer_azimuth"].to_numpy(),
            )

        return L1bImage(result)
