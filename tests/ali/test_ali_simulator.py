from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sasktran2 as sk
from aliprocessing.geometry import InstrumentFrame, sun_direction, tangent_geometry
from aliprocessing.l2.optical import aerosol_median_radius_db
from aliprocessing.solar import ChannelSolarIrradiance

from hawcsimulator.ali import ALISimulator
from hawcsimulator.geometry.observation import observation_geometry
from hawcsimulator.geometry.orbit import NOMINAL_ORBIT, time_at_tangent_latitude

FORCED = {
    "tangent_latitude": 30.0,
    "tangent_longitude": 0.0,
    "tangent_solar_zenith_angle": 60.0,
    "tangent_solar_azimuth_angle": 120.0,
    "time": pd.Timestamp("2022-01-01T12:00:00Z"),
}

# A sample from the example HAWCsat orbit (along=2650, across=256), as in
# tests/test_limb_observation.py
ORBIT = {
    "tangent_latitude": 30.3662,
    "tangent_longitude": 178.1106,
    "observer_latitude": 9.9504,
    "observer_longitude": 175.1511,
    "time": pd.Timestamp("2019-08-01T01:54:28"),
}

ALTITUDE_GRID = np.arange(0.0, 65001.0, 1000.0)
TRUE_EXTINCTION = 1e-8 + 2e-6 * np.exp(-0.5 * ((ALTITUDE_GRID - 20e3) / 4e3) ** 2)
TRUE_MEDIAN_RADIUS = 80.0


def _aerosol():
    return sk.constituent.ExtinctionScatterer(
        aerosol_median_radius_db(),
        ALTITUDE_GRID,
        TRUE_EXTINCTION,
        745.0,
        "extend",
        median_radius=np.full(len(ALTITUDE_GRID), TRUE_MEDIAN_RADIUS),
    )


def _boresight_tangent_point(pointing, time):
    frame = InstrumentFrame.from_attitude(
        pointing["spacecraft_latitude"],
        pointing["spacecraft_longitude"],
        pointing["spacecraft_altitude"],
        pointing["boresight_azimuth"],
        pointing["boresight_elevation"],
        pointing["boresight_roll"],
    )
    return tangent_geometry(
        frame.observer, frame.boresight[np.newaxis], sun_direction(time)
    )


def test_boresight_looks_at_the_tangent_point():
    data = ALISimulator().run(["pointing"], FORCED)

    boresight = _boresight_tangent_point(data["pointing"], np.datetime64("2022-01-01"))
    # The nominal acquisition points the boresight at 25 km
    assert float(boresight.altitude[0]) == pytest.approx(25000.0, abs=100.0)
    assert float(boresight.latitude[0]) == pytest.approx(30.0, abs=0.1)
    assert float(boresight.longitude[0]) == pytest.approx(0.0, abs=0.1)


def test_acquisition_pattern_sets_the_pointing_and_timing():
    data = ALISimulator().run(
        ["pointing", "l1a"],
        {
            **FORCED,
            "acquisition_cfg": {
                "pointing": {"boresight_tangent_altitude_m": 27000.0},
                "timing": {"num_sets": 3, "exposure": 0.5, "filter_change_s": 0.2},
            },
            "l1a_cfg": {"noise": False},
        },
    )

    boresight = _boresight_tangent_point(data["pointing"], np.datetime64("2022-01-01"))
    assert float(boresight.altitude[0]) == pytest.approx(27000.0, abs=100.0)

    l1a = data["l1a"]
    # 3 sets of 5 channels, each set ending with a blank frame
    assert l1a.sizes["time"] == 18
    frames = l1a.to_dataset() if hasattr(l1a, "to_dataset") else l1a
    elevation = frames["boresight_elevation_angle"].to_numpy()
    # Every set points the same way
    assert len(np.unique(np.round(elevation, 6))) == 1
    # Each 0.5 s frame averages 6 exposures of 83.3 ms, each followed by the
    # detector's 16.7 ms dead time, then the filter wheel moves for 0.2 s
    np.testing.assert_allclose(frames["exposure_time"], 0.5 / 6)
    np.testing.assert_array_equal(frames["num_averaged_exposures"], 6)
    step = np.diff(frames["time"].to_numpy()) / np.timedelta64(1, "s")
    np.testing.assert_allclose(step, 6 * (0.5 / 6 + 0.0167) + 0.2, atol=1e-6)
    assert l1a.attrs["hawcsimulator_acquisition"] == "nominal"


def test_default_tangent_altitudes_cover_the_l1b_grid():
    simulator = ALISimulator()
    cal_db = simulator.run(["calibration_database"])["calibration_database"]

    tangent_altitudes = simulator._initialize_data()["viewing_tangent_altitudes"]
    half_width = float(cal_db["altitude_kernel_half_width"])
    assert tangent_altitudes.min() <= float(cal_db["altitude"].min()) - half_width
    assert tangent_altitudes.max() >= float(cal_db["altitude"].max()) + half_width


def test_radiance_uses_the_channel_solar_irradiance():
    # Even when the user's constituents replace the simulator's defaults: L2 models
    # radiance with the band-averaged solar irradiance, and a different solar
    # spectrum biases its extinction and particle size
    data = ALISimulator().run(
        ["sk2_atmosphere"], {**FORCED, "constituents": {"aerosol": _aerosol()}}
    )

    assert isinstance(
        data["sk2_atmosphere"]["solar_irradiance"], ChannelSolarIrradiance
    )


def test_forced_solar_angles_replace_the_l1b_solar_angles():
    data = ALISimulator().run(["l1b"], {**FORCED, "l1a_cfg": {"noise": False}})

    np.testing.assert_allclose(data["l1b"]["solar_zenith_angle"], 60.0)
    np.testing.assert_allclose(data["l1b"]["relative_solar_azimuth_angle"], 120.0)
    assert data["l1b"].attrs["hawcsimulator_instrument"] == "ALI-0-FWHEEL-REP1"
    assert data["l1b"].attrs["hawcsimulator_processing"] == "v0.1"
    assert data["l1b"].attrs["hawcsimulator_acquisition"] == "nominal"


def test_frames_average_exposures_within_the_detector_limit():
    l1a = ALISimulator().run(["l1a"], {**FORCED, "l1a_cfg": {"noise": False}})["l1a"]

    assert np.all(l1a["exposure_time"] <= 0.0925)
    assert np.all(l1a["num_averaged_exposures"] > 1)


def test_time_based_l1b_solar_zenith_angle_matches_the_radiance():
    data = ALISimulator().run(
        ["l1b", "observation"], {**ORBIT, "l1a_cfg": {"noise": False}}
    )

    l1b = data["l1b"]
    boresight_bin = l1b.isel(across_track=l1b.sizes["across_track"] // 2)
    l1b_sza = float(boresight_bin["solar_zenith_angle"].mean())
    cos_sza = data["observation"].observation.reference_cos_sza()["measurement"]
    assert l1b_sza == pytest.approx(float(np.rad2deg(np.arccos(cos_sza))), abs=0.5)


def test_orbit_l1b_geometry_matches_the_observation():
    # L1A to L1B geolocates each frame with aliprocessing's own geometry and sun
    time = time_at_tangent_latitude(NOMINAL_ORBIT, 45.0, np.datetime64("2026-06-21"))
    data = ALISimulator().run(
        ["l1b", "observation"],
        {"time": time, "l1a_cfg": {"noise": False}},
        config={"observation_method": "orbit"},
    )

    l1b = data["l1b"].isel(observation_set=0)
    # The across-track bin centred on the boresight, or the two either side of it
    angle = np.abs(l1b["across_track_angle"].to_numpy())
    centre = np.flatnonzero(np.isclose(angle, angle.min()))
    boresight = l1b.isel(across_track=centre).mean(["across_track", "wavelength"])

    geometry = observation_geometry(data["observation"].observation)
    reference_altitude = l1b["reference_altitude"].to_numpy()
    for name, atol in [
        ("solar_zenith_angle", 0.02),
        ("relative_solar_azimuth_angle", 0.02),
        ("tangent_latitude", 0.001),
    ]:
        expected = np.interp(
            reference_altitude, geometry["tangent_altitude"], geometry[name]
        )
        np.testing.assert_allclose(boresight[name], expected, atol=atol)


def _assert_retrieves_the_truth(l2):
    # One profile: the first observation set's middle across-track bin
    l2 = l2.isel(observation_set=0, across_track=0)
    altitudes = [15e3, 20e3, 25e3, 30e3]
    extinction = l2["aerosol_extinction"].interp(altitude=altitudes)
    truth = np.interp(altitudes, ALTITUDE_GRID, TRUE_EXTINCTION)

    np.testing.assert_allclose(extinction, truth, rtol=0.05)
    median_radius = float(l2["aerosol_median_radius"].interp(altitude=20e3))
    assert median_radius == pytest.approx(TRUE_MEDIAN_RADIUS, rel=0.05)


@pytest.mark.slow
def test_noise_free_retrieval_returns_the_truth():
    data = ALISimulator().run(
        ["l2", "l2_truth"],
        {
            **FORCED,
            "constituents": {"aerosol": _aerosol()},
            "l1a_cfg": {"noise": False},
            "l2_cfg": {"minimizer": {"verbose": 0}},
        },
    )
    # l2_truth is the truth on the L2's grid
    l2_truth = data["l2_truth"].isel(observation_set=0, across_track=0)
    np.testing.assert_allclose(
        l2_truth["aerosol_extinction"].interp(altitude=20e3),
        np.interp(20e3, ALTITUDE_GRID, TRUE_EXTINCTION),
        rtol=1e-6,
    )

    _assert_retrieves_the_truth(data["l2"])


@pytest.mark.slow
def test_noise_free_retrieval_on_the_orbit_returns_the_truth():
    # Forward scatter at 45 N, with the solar angles varying along the lines of sight
    time = time_at_tangent_latitude(NOMINAL_ORBIT, 45.0, np.datetime64("2026-06-21"))
    data = ALISimulator().run(
        ["l2"],
        {
            "time": time,
            "constituents": {"aerosol": _aerosol()},
            "l1a_cfg": {"noise": False},
            "l2_cfg": {"minimizer": {"verbose": 0}},
        },
        config={"observation_method": "orbit"},
    )

    _assert_retrieves_the_truth(data["l2"])


@pytest.mark.slow
def test_noisy_retrieval():
    data = ALISimulator().run(
        ["l2"],
        {
            **ORBIT,
            "constituents": {"aerosol": _aerosol()},
            "l1a_cfg": {"seed": 1},
            "l2_cfg": {"minimizer": {"verbose": 0}},
        },
    )

    l2 = data["l2"]
    assert l2.sizes["observation_set"] == 1
    assert l2.sizes["across_track"] == 1
    assert np.all(np.isfinite(l2["aerosol_extinction"]))
    assert np.all(l2["aerosol_extinction_noise_error"] > 0)
