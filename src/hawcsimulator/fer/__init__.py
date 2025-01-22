from __future__ import annotations

import numpy as np
import sasktran2 as sk
import xarray as xr
from skretrieval.core.sasktranformat import SASKTRANRadiance
from skretrieval.retrieval.forwardmodel import IdealViewingMixin
from skretrieval.retrieval.observation import Observation


class FERGenerator:
    def run(self):
        pass


class FERGeneratorBasic(IdealViewingMixin):
    def __init__(self, observation: Observation, model_altitude_grid: np.array):
        IdealViewingMixin.__init__(self, observation, model_altitude_grid)

        self._viewing_geo = self._construct_viewing_geo()
        self._model_geo = {
            "measurement": self._viewing_geo["measurement"].model_geometry(
                model_altitude_grid
            )
        }

        self._sk_config = sk.Config()

    @property
    def model_geo(self):
        return self._model_geo["measurement"]

    @property
    def viewing_geo(self):
        return self._viewing_geo["measurement"]

    @property
    def sk_config(self):
        return self._sk_config

    def run(self, atmosphere: sk.Atmosphere):
        self.model_geo.refractive_index = (
            sk.optical.refraction.ciddor_index_of_refraction(
                atmosphere.temperature_k, atmosphere.pressure_pa, 0.0, 400, 1350.0
            )
        )

        engine = sk.Engine(self._sk_config, self.model_geo, self.viewing_geo)

        sk2_rad = engine.calculate_radiance(atmosphere)

        return SASKTRANRadiance.from_sasktran2(sk2_rad)
