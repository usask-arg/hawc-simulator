from __future__ import annotations

import numpy as np
import pandas as pd
import sasktran2 as sk
import xarray as xr
from showlib.por.l2.merra import get_tropopause_height


def por_from_atmosphere(atmo: sk.Atmosphere, time: pd.Timestamp):
    alt_grid = atmo.model_geometry.altitudes()
    por = xr.Dataset(
        {
            "temperature": (["altitude"], atmo.temperature_k),
            "pressure": (["altitude"], atmo.pressure_pa),
            "tropopause_altitude": get_tropopause_height(alt_grid, atmo.temperature_k),
        },
        coords={"altitude": alt_grid},
    )

    por = xr.concat([por], dim="time")
    por.coords["time"] = [np.datetime64(time)]

    return por
