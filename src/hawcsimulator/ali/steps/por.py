from __future__ import annotations

import sasktran2 as sk
import xarray as xr

from hawcsimulator.ali.por import por_from_atmosphere
from hawcsimulator.datastructures.viewinggeo import ObservationContainer


def program_of_record(
    sk2_atmosphere: sk.Atmosphere, observation: ObservationContainer
) -> xr.Dataset:
    return por_from_atmosphere(sk2_atmosphere, observation.time)
