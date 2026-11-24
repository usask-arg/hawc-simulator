from __future__ import annotations

import numpy as np
import pandas as pd
import sasktran2 as sk
import xarray as xr
from aliprocessing.l2.optical import aerosol_median_radius_db

from hawcsimulator.steps.truth import on_l2_grid

FORCED = {
    "tangent_latitude": 30.0,
    "tangent_longitude": 0.0,
    "tangent_solar_zenith_angle": 60.0,
    "tangent_solar_azimuth_angle": 120.0,
    "time": pd.Timestamp("2022-01-01T12:00:00Z"),
}


def test_truth_is_the_simulated_state(tmp_path):
    from hawcsimulator.ali import ALISimulator

    altitude = np.arange(0.0, 65001.0, 1000.0)
    extinction = 1e-8 + 1e-6 * np.exp(-0.5 * ((altitude - 18e3) / 3e3) ** 2)
    aerosol = sk.constituent.ExtinctionScatterer(
        aerosol_median_radius_db(),
        altitude,
        extinction,
        745.0,
        "extend",
        median_radius=np.full(len(altitude), 80.0),
    )
    data = ALISimulator().run(
        ["truth", "program_of_record"],
        {**FORCED, "constituents": {"aerosol": aerosol}},
    )
    truth = data["truth"]

    grid = truth["altitude"].to_numpy()
    np.testing.assert_allclose(
        truth["aerosol_extinction"], np.interp(grid, altitude, extinction, right=0)
    )
    assert truth["aerosol_extinction"].attrs["wavelength_nm"] == 745.0
    np.testing.assert_allclose(truth["aerosol_median_radius"], 80.0)
    assert float(truth["surface_albedo"]) == 0.3
    np.testing.assert_allclose(
        truth["temperature"], data["program_of_record"]["temperature"].isel(time=0)
    )
    assert abs(truth.attrs["tangent_latitude"] - 30.0) < 0.1

    # It saves as netCDF
    truth.to_netcdf(tmp_path / "truth.nc")
    xr.testing.assert_allclose(xr.load_dataset(tmp_path / "truth.nc"), truth)


def test_show_truth_includes_the_gases_the_radiance_used():
    from hawcsimulator.show import SHOWSimulator

    truth = SHOWSimulator().run(["truth"], FORCED)["truth"]

    # CH4 is added by the front end radiance step, not the atmosphere
    for name in ("h2o_vmr", "o3_vmr", "ch4_vmr", "pressure", "tropopause_altitude"):
        assert name in truth


def test_truth_on_the_l2_grid():
    truth = xr.Dataset(
        {
            "aerosol_extinction": (
                "altitude",
                np.linspace(0, 1e-6, 66),
                {"wavelength_nm": 745.0},
            ),
            "surface_albedo": ((), 0.3),
        },
        coords={"altitude": np.arange(0.0, 65001.0, 1000.0)},
    )
    like = xr.DataArray(
        np.zeros((2, 3, 5)),
        dims=("observation_set", "across_track", "altitude"),
        coords={"altitude": [10e3, 12.5e3, 15e3, 17.5e3, 20e3]},
    )
    names = {
        "aerosol_extinction": "aerosol_extinction",
        "surface_albedo": "surface_albedo",
    }

    on_grid = on_l2_grid(truth, names, like, 745.0)
    assert on_grid["aerosol_extinction"].dims == like.dims
    np.testing.assert_allclose(
        on_grid["aerosol_extinction"].isel(observation_set=1, across_track=2),
        np.interp(like["altitude"], truth["altitude"], truth["aerosol_extinction"]),
    )
    assert on_grid["surface_albedo"].dims == ("observation_set", "across_track")

    # Extinction at another wavelength than the L2's is left out
    assert "aerosol_extinction" not in on_l2_grid(truth, names, like, 1350.0)
