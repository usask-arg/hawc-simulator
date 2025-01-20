from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import appdirs
import numpy as np
import xarray as xr


def _generate_ideal_spectrograph_cal_db():
    res = xr.Dataset()

    res.coords["sample_wavelengths"] = np.arange(400, 800, 1)

    return res

def _generate_ideal_spex_cal_db():
    return xr.Dataset()


def calibration_database(name: str = "ideal", version: str = "v1"):
    dir = (
        Path(
            appdirs.AppDirs(
                appname="hawc-simulator", appauthor="usask-arg"
            ).user_data_dir
        )
        / "ali"
        / "calibration"
    )

    file = dir / f"{name}_{version}.nc"

    if not file.exists() or True:
        if name == "ideal_spectrograph":
            cal_db = _generate_ideal_spectrograph_cal_db()


            file.parent.mkdir(parents=True, exist_ok=True)
            cal_db.to_netcdf(file)

            return file

        if name == "ideal_spex":
            cal_db = _generate_ideal_spex_cal_db()


            file.parent.mkdir(parents=True, exist_ok=True)
            cal_db.to_netcdf(file)

            return file

        error_message = f"Unknown calibration database name: {name}"
        raise ValueError(error_message)
    return file
