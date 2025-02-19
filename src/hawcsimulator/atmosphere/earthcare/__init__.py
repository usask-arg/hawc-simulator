from __future__ import annotations

import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sasktran2 as sk
import xarray as xr
from netCDF4 import Dataset
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import Delaunay
from skretrieval.core.lineshape import Rectangle

from hawcsimulator.appconfig import load_user_config


def _earthcare_folder():
    cfg = load_user_config()

    if "earthcare_folder" in cfg:
        return Path(cfg["earthcare_folder"]).expanduser()
    msg = "No earthcare folder specified the user config. Add 'earthcare_folder' to the user config."
    raise ValueError(msg)


def _latitude_average(
    central_latitude: float, data: xr.Dataset, lat_range=1
) -> xr.Dataset:
    if (
        central_latitude < data["latitude"].min()
        or central_latitude > data["latitude"].max()
    ):
        msg = f"Central latitude {central_latitude} is outside the range of the data"
        raise ValueError(msg)

    lat_range = (central_latitude - lat_range, central_latitude + lat_range)
    data = data.where(data["latitude"] > lat_range[0], drop=True)
    data = data.where(data["latitude"] < lat_range[1], drop=True)

    return data.mean(dim="nx")


def _to_uniform_spacing(data: xr.Dataset, spacing: float, out_var: str) -> xr.Dataset:
    out_grid = np.arange(0, data["height"].max() * 1000 + spacing, spacing)

    ls = Rectangle(width=spacing)

    transform = np.zeros((len(out_grid), len(data["height"])))

    for i, a in enumerate(out_grid):
        transform[i] = ls.integration_weights(a, data["height"] * 1000)

    return (
        xr.DataArray(transform, dims=["altitude", "nz"], coords={"altitude": out_grid})
        @ data[out_var]
    )


def _load_lat_lon(data: xr.Dataset):
    ds = xr.open_dataset(_earthcare_folder() / "scene_ext_3d-1-0.680.nc")

    data["latitude"] = ds["latitude"]
    data["longitude"] = ds["longitude"]

    if "nz" in data.dims:
        data["height"] = ds["height"]

    return data


def _brdf(central_latitude: float):
    # brdf_geo = xr.open_dataset(_earthcare_folder() / "Test_data_39316D2_2014120712_BRDF_geo.nc").rename({"x": "nx"})
    brdf_iso = xr.open_dataset(
        _earthcare_folder() / "Test_data_39316D2_2014120712_BRDF_iso.nc"
    ).rename({"x": "nx"})
    brdf_vol = xr.open_dataset(
        _earthcare_folder() / "Test_data_39316D2_2014120712_BRDF_vol.nc"
    ).rename({"x": "nx"})

    # brdf_geo = _load_lat_lon(brdf_geo)
    brdf_iso = _load_lat_lon(brdf_iso)
    brdf_vol = _load_lat_lon(brdf_vol)

    # brdf_geo = _latitude_average(central_latitude, brdf_geo)
    brdf_iso = _latitude_average(central_latitude, brdf_iso)
    brdf_vol = _latitude_average(central_latitude, brdf_vol)

    # This isn't really correct, but we will do it this way anyways
    band_wavel = [500.0, 645.0, 860.0, 1640.0]
    band_idx = [0, 2, 3, 5]

    # brdf_geo = brdf_geo.isel(band=band_idx)
    brdf_iso = brdf_iso.isel(band=band_idx)
    brdf_vol = brdf_vol.isel(band=band_idx)

    return xr.Dataset(
        {  # "geo": (["wavelength"], brdf_geo["BRDF_geo"].values),
            "iso": (["wavelength"], brdf_iso["BRDF_iso"].values),
            "vol": (["wavelength"], brdf_vol["BRDF_vol"].values),
        },
        coords={"wavelength": band_wavel},
    )


def _aerosol(latitude: float, type_id: int):
    vars = {"ext": "Extinction", "w": "SS_alb", "g": "g"}

    result = {}
    for k, v in vars.items():
        temp = []
        files = _earthcare_folder().glob(f"scene_{k}_3d-{type_id}-*.nc")
        for f in files:
            vw = float(re.search(r"[\d\.]+\.nc", f.stem + ".nc").group(0)[:-3]) * 1000

            aerosol = xr.open_dataset(f)
            aerosol[v].values[aerosol[v].values < 0] = 0

            temp.append(_latitude_average(latitude, aerosol))
            temp[-1]["wavelength"] = vw
            temp[-1] = temp[-1].set_coords("wavelength")
            temp[-1][v] = temp[-1][v].expand_dims("wavelength")

        result[k] = xr.concat(temp, dim="wavelength", data_vars="minimal")

    ext = _to_uniform_spacing(result["ext"], 1000, "Extinction")
    result["w"]["scat_ext"] = result["ext"]["Extinction"] * (result["w"]["SS_alb"])
    scat_ext = _to_uniform_spacing(result["w"], 1000, "scat_ext")
    g = _to_uniform_spacing(result["g"], 1000, "g")

    result = xr.Dataset()
    result["k"] = ext
    result["w"] = scat_ext / ext
    result["g"] = g

    return result


def _water(central_latitude: float):
    water = xr.open_dataset(
        _earthcare_folder() / "Test_data_39316D2_2014120712_specific_humidity.nc"
    )

    water = _load_lat_lon(water)

    water["specific_humidity"].values[water["specific_humidity"].values < 0] = 0
    water["specific_humidity"].values[np.isnan(water["specific_humidity"].values)] = 0

    water = _latitude_average(central_latitude, water)
    return (
        _to_uniform_spacing(water, 1000, "specific_humidity") * 28.9647 / 18.02 / 1000
    )


def _temperature(central_latitude: float):
    temperature = xr.open_dataset(
        _earthcare_folder() / "Test_data_39316D2_2014120712_temperature.nc"
    )

    temperature = _load_lat_lon(temperature)

    temperature["specific_humidity"].values[
        temperature["specific_humidity"].values < 0
    ] = 0
    temperature["specific_humidity"].values[
        np.isnan(temperature["specific_humidity"].values)
    ] = 0

    temperature = _latitude_average(central_latitude, temperature)
    temperature = _to_uniform_spacing(temperature, 1000, "specific_humidity")


if __name__ == "__main__":
    _aerosol(30.0, 7)
