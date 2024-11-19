from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from showlib.l1b.data import L1bDataSet, L1bImage
from skretrieval.core.lineshape import UserLineShape
from skretrieval.retrieval.forwardmodel import SpectrometerMixin
from skretrieval.retrieval.measvec import MeasurementVector, select


class L1bGenerator:
    def run(self, fer: xr.Dataset):
        pass


class L1bGeneratorILS(L1bGenerator, SpectrometerMixin):
    def __init__(self, cal_db: xr.Dataset, observation, noise_model=None, **kwargs):
        self._ils = lambda w: UserLineShape(
            cal_db.hires_wavenumber.to_numpy(),
            cal_db.sel(sample_wavenumber=1e7 / w, method="nearest")["ils"].to_numpy(),
            False,
        )
        self._observation = observation

        self._meas_vec = {
            "*": MeasurementVector(
                lambda l1, ctxt, **kwargs: select(l1, **kwargs)  # noqa: ARG005
            )
        }

        SpectrometerMixin.__init__(
            self, self._ils, spectral_native_coordinate="wavenumber_cminv", **kwargs
        )

        self._inst_model = self._construct_inst_model()

        if noise_model is not None:
            self._noise_model = noise_model
        else:
            self._noise_model = lambda rad: rad * 0.01

    def run(self, fer: xr.Dataset):
        result = self._inst_model["measurement"].model_radiance(fer, None)

        num_los = len(result.data["tangent_altitude"].to_numpy())

        l1b = L1bImage.from_np_arrays(
            result.data["radiance"].to_numpy()[::-1, :],
            self._noise_model(result.data["radiance"]).to_numpy()[::-1, :],
            result.data["tangent_altitude"].to_numpy(),
            result.data["tangent_latitude"].to_numpy(),
            result.data["tangent_longitude"].to_numpy(),
            np.ones(num_los) * result.data["wavenumber"].to_numpy()[-1],
            np.ones(num_los)
            * (
                result.data["wavenumber"].to_numpy()[0]
                - result.data["wavenumber"].to_numpy()[1]
            ),
            pd.to_datetime("2021-01-01"),
            0.0,
            0.0,
            float(result.data["observer_altitude"].to_numpy()[0]),
            np.rad2deg(np.arccos(result.data["tangent_cos_sza"].to_numpy())),
            result.data["tangent_solar_azimuth"].to_numpy(),
            result.data["tangent_observer_azimuth"].to_numpy(),
        )

        return L1bDataSet.from_image(l1b)
