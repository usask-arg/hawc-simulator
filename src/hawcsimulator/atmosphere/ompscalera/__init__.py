from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import sasktran2 as sk
import xarray as xr
from skretrieval.core.lineshape import Rectangle

from hawcsimulator.appconfig import load_user_config


def _curtain_file():
    cfg = load_user_config()

    if "omps_calipso_era5_curtain" in cfg:
        return Path(cfg["omps_calipso_era5_curtain"]).expanduser()
    msg = "No curtain file specified in the user config. Add 'omps_calipso_era5_curtain' to the user config."
    raise ValueError(msg)


def _to_uniform_spacing(
    data: xr.Dataset, spacing: float, out_grid: np.ndarray, out_var: str
) -> xr.Dataset:
    ls = Rectangle(width=spacing)

    transform = np.zeros((len(out_grid), len(data["altitude"])))

    for i, a in enumerate(out_grid):
        transform[i] = ls.integration_weights(a, data["altitude"])

    return (
        xr.DataArray(
            transform,
            dims=["altitude2", "altitude"],
            coords={"altitude2": out_grid, "altitude": data.altitude},
        )
        @ data[out_var]
    ).rename({"altitude2": "altitude"})


def load_data(central_latitude):
    ds = xr.open_datatree(_curtain_file())

    calipso = xr.Dataset(ds["CALIPSO"])
    omps = xr.Dataset(ds["OMPS"])
    era5 = xr.Dataset(ds["ERA5"])

    omps = omps.swap_dims({"time": "latitude"}).interp(latitude=era5.latitude)
    era5 = era5.swap_dims({"time": "latitude"}).fillna(0.0)
    calipso = calipso.swap_dims({"time": "latitude"})

    omps = (
        omps.interp(altitude=np.arange(0, 40.0, 0.5))
        .swap_dims({"time": "latitude"})
        .interp(latitude=central_latitude)
    )

    omps["h2o_vmr"] = (
        _to_uniform_spacing(era5, 0.5, omps.altitude.to_numpy(), "specific_humidity")
        * 28.97
        / 18.01528
    ).interp(latitude=central_latitude)

    return omps


if __name__ == "__main__":
    load_data(30.0)
