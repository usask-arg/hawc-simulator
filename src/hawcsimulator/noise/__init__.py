from __future__ import annotations

import abc

import numpy as np
import xarray as xr


class NoiseModel(abc.ABC):
    @abc.abstractmethod
    def calc_noise(self, signal: xr.DataArray):
        pass


class ConstantNoise(NoiseModel):
    def __init__(self, noise_level: float):
        self._noise_level = noise_level

    def calc_noise(self, signal: xr.DataArray):

        noise_sigma = signal * self._noise_level

        add_noise = np.random.default_rng().normal(0, noise_sigma.to_numpy())

        return (signal + add_noise, noise_sigma)
