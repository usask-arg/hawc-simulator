from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from aliprocessing.l1b.data import L1bSpectra, L1bImage
from skretrieval.core.lineshape import UserLineShape, Gaussian
from skretrieval.retrieval.forwardmodel import SpectrometerMixin
from skretrieval.retrieval.measvec import MeasurementVector, select
from scipy.interpolate import CubicSpline


class L1bGenerator:
    def run(self, fer: xr.Dataset):
        pass


class L1bGeneratorIdeal(L1bGenerator):
    def __init__(self, cal_db: xr.Dataset, observation, pol_states, noise_model=None, include_noise=False, **kwargs):
        self._observation = observation

        self._pol_states = pol_states

        self._dolp_error = kwargs.get("dolp_error", 0.003)
        self._aolp_error = np.deg2rad(kwargs.get("aolp_error", 0.2))
        self._intensity_error = kwargs.get("intensity_error", 0.01)
        self._noise = include_noise

    def run(self, fer: xr.Dataset):
        result = {}

        I = fer.data["radiance"].isel(stokes=0)


        result["I"] = L1bSpectra.from_np_arrays(
            I.to_numpy(),
            np.abs(I.to_numpy() * self._intensity_error),
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
            dolp = np.sqrt(fer.data["radiance"].isel(stokes=1)**2 + fer.data["radiance"].isel(stokes=2)**2) / fer.data["radiance"].isel(stokes=0)
            result["dolp"] = L1bSpectra.from_np_arrays(
                dolp.to_numpy(),
                np.zeros_like(dolp.to_numpy()) + self._dolp_error,
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

        if "aolp" in self._pol_states:
            aolp = 0.5 * np.arctan(fer.data["radiance"].isel(stokes=2) / fer.data["radiance"].isel(stokes=1))
            result["aolp"] = L1bSpectra.from_np_arrays(
                aolp.to_numpy(),
                np.zeros_like(aolp.to_numpy()) + self._aolp_error,
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
        if "q" in self._pol_states:
            q = fer.data["radiance"].isel(stokes=1) / fer.data["radiance"].isel(stokes=0)

            aolp = 0.5 * np.arctan(fer.data["radiance"].isel(stokes=2) / fer.data["radiance"].isel(stokes=1))
            dolp = np.sqrt(fer.data["radiance"].isel(stokes=1)**2 + fer.data["radiance"].isel(stokes=2)**2) / fer.data["radiance"].isel(stokes=0)

            # Estimate the error in q from the errors in DOLP and AOLP
            # q = dolp * cos(2*aolp)

            # abs_error = self._dolp_error * np.abs(np.cos(2*aolp))

            # abs errors from aolp are
            abs_error = dolp * np.abs(-2 * np.sin(2*aolp) * self._aolp_error)
            abs_error += self._dolp_error

            result["q"] = L1bSpectra.from_np_arrays(
                q.to_numpy(),
                np.zeros_like(q.to_numpy()) + abs_error.to_numpy(),
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

        if self._noise:
            for k in result:
                result[k]._ds["radiance"] += np.random.normal(0, result[k]._ds["radiance_noise"].to_numpy())

        return L1bImage(result)


class L1bGeneratorILS(L1bGenerator, SpectrometerMixin):
    def __init__(self, cal_db: xr.Dataset, observation, noise_model=None, **kwargs):
        self._ils = lambda w: Gaussian(fwhm=2)

        self._observation = observation

        self._meas_vec = {
            "*": MeasurementVector(
                lambda l1, ctxt, **kwargs: select(l1, **kwargs)  # noqa: ARG005
            )
        }

        stokes_sensitivities = {
            "plus_modulation": [0.5, 0.5, 0,0],
            "minus_modulation": [0.5, -0.5, 0,0],
        }

        SpectrometerMixin.__init__(
            self, self._ils, spectral_native_coordinate="wavelength_nm", stokes_sensitivities=stokes_sensitivities, **kwargs
        )

        self._inst_model = self._construct_inst_model()

        if noise_model is not None:
            self._noise_model = noise_model
        else:
            self._noise_model = lambda rad: rad * 0.01

        self._include_modulation = bool(cal_db["include_modulation"])

    def run(self, fer: xr.Dataset):
        # Adjust the Q component of the FER to be the modulated component

        result = {}

        if self._include_modulation:
            # This is DOLP * I
            dolp = np.sqrt(fer.data["radiance"].isel(stokes=1)**2 + fer.data["radiance"].isel(stokes=2)**2)
            aolp = 0.5 * np.arctan(fer.data["radiance"].isel(stokes=2) / fer.data["radiance"].isel(stokes=1))

            modulation = dolp * np.cos(2*np.pi * 13000 / fer.data["wavelength_nm"] + 2.0*aolp)

            original_rad = fer.data["radiance"].copy()

            fer.data["radiance"].to_numpy()[:, :, 1] = modulation
            inst_result = self._inst_model["measurement"].model_radiance(fer, None)

            # Reset the original radiance
            fer.data["radiance"] = original_rad
        else:
            inst_result = self._inst_model["measurement"].model_radiance(fer, None)

        # Always include I
        result["I"] = inst_result["plus_modulation"].data.radiance + inst_result["minus_modulation"].data.radiance


        if self._include_modulation:
            instrument_modulation = inst_result["plus_modulation"].data["radiance"] - inst_result["minus_modulation"].data["radiance"]

            highres_wavel = np.arange(instrument_modulation.wavelength.min(), instrument_modulation.wavelength.max(), 0.01)

            all_aolp = np.zeros((len(instrument_modulation.los), len(instrument_modulation.wavelength)))
            all_dolp = np.zeros((len(instrument_modulation.los), len(instrument_modulation.wavelength)))

            for i in range(len(instrument_modulation.los)):
                modulation = instrument_modulation.isel(los=i).to_numpy()
                # Interpolate to a highres grid
                spline = CubicSpline(instrument_modulation.wavelength.to_numpy(), modulation)
                interp_modulation = spline(highres_wavel)
                interp_deriv = spline(highres_wavel, 1)

                zero_crossings = np.where(np.diff(np.signbit(interp_modulation)))[0]
                aolp_wavelength = highres_wavel[zero_crossings]

                # 2 * (aolp + pi * 13000 / lambda) = n * pi/2
                # aolp = n * pi/4 - pi * 13000 / lambda, pick n so we are in the range 0 to pi
                a = - np.pi * 13000 / aolp_wavelength

                # Put in range 0 to pi
                a = a % np.pi

                all_aolp[i, :] = np.interp(instrument_modulation.wavelength.to_numpy(), aolp_wavelength, a, left=np.nan, right=np.nan)

                # Zeros of deriv at equal to DOLP
                zero_crossings = np.where(np.diff(np.signbit(interp_deriv)))[0]
                dolp_wavelength = highres_wavel[zero_crossings]
                d = np.abs(spline(dolp_wavelength))

                all_dolp[i, :] = np.interp(instrument_modulation.wavelength.to_numpy(), dolp_wavelength, d, left=np.nan, right=np.nan)

                pass

            # Zero crossings are where AOLP = -pi * 13000 / lambda

            pass

        spectra = {}
        for k, v in result.items():
            spectra[k] = L1bSpectra.from_np_arrays(
                v.data["radiance"].to_numpy(),
                self._noise_model(v.data["radiance"]).to_numpy(),
                v.data["tangent_altitude"].to_numpy(),
                v.data["tangent_latitude"].to_numpy(),
                v.data["tangent_longitude"].to_numpy(),
                v.data["wavelength"].to_numpy(),
                pd.to_datetime("2021-01-01"),
                0.0,
                0.0,
                float(v.data["observer_altitude"].to_numpy()[0]),
                np.rad2deg(np.arccos(v.data["tangent_cos_sza"].to_numpy())),
                v.data["tangent_solar_azimuth"].to_numpy(),
                v.data["tangent_observer_azimuth"].to_numpy(),
            )

        # Create L1b data here
        return L1bImage(spectra)
