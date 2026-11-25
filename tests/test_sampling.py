from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hawcsimulator.ali import ALISimulator
from hawcsimulator.geometry.observation import observation_geometry
from hawcsimulator.show import SHOWSimulator
from hawcsimulator.steps.limb_observation import observation__orbit

PERIOD = {
    "sampling_start": pd.Timestamp("2026-06-21T00:00:00Z"),
    "sampling_end": pd.Timestamp("2026-06-21T03:00:00Z"),
}
FORCED = {
    "tangent_latitude": 30.0,
    "tangent_longitude": 0.0,
    "tangent_solar_zenith_angle": 60.0,
    "tangent_solar_azimuth_angle": 120.0,
    "time": pd.Timestamp("2022-01-01T12:00:00Z"),
}


@pytest.mark.parametrize(
    ("simulator", "interval"), [(ALISimulator, 20.0), (SHOWSimulator, 2.4)]
)
def test_observations_start_every_sampling_interval_in_daylight(simulator, interval):
    sampling = simulator().run(["sampling"], PERIOD)["sampling"]

    assert sampling.attrs["sampling_interval_s"] == interval
    assert sampling.attrs["hawcsimulator_acquisition"] == "nominal"
    seconds = (sampling["time"] - sampling["time"][0]) / np.timedelta64(1, "s")
    np.testing.assert_allclose(seconds / interval, np.round(seconds / interval))
    # One interval apart, except across the dark part of an orbit
    steps = np.diff(seconds.to_numpy())
    dark = steps > 10 * interval
    np.testing.assert_allclose(steps[~dark], interval)
    assert np.any(dark)
    assert np.all(steps[dark] > 20 * 60)
    assert np.all(sampling["solar_zenith_angle"] <= 88.0)


def test_sampling_is_the_orbit_observation_geometry():
    sampling = SHOWSimulator().run(["sampling"], PERIOD)["sampling"]

    times = sampling["time"].to_numpy()
    for i in (0, len(times) // 2, -1):
        sample = sampling.isel(time=i)
        observation = observation__orbit(
            viewing_tangent_altitudes=np.array([20e3, 25e3, 30e3]),
            time=pd.Timestamp(times[i], tz="UTC"),
            sample_wavelengths=np.array([745.0]),
        )
        geometry = observation_geometry(observation.observation)
        for name in (
            "tangent_latitude",
            "tangent_longitude",
            "solar_zenith_angle",
            "relative_solar_azimuth_angle",
            "los_azimuth_angle",
        ):
            assert float(sample[name]) == pytest.approx(geometry[name][1], abs=1e-6)
        assert float(sample["spacecraft_altitude"]) == pytest.approx(
            geometry["observer_altitude"][1]
        )


def test_ascending_follows_the_tangent_point():
    sampling = SHOWSimulator().run(["sampling"], PERIOD)["sampling"]

    latitude = sampling["tangent_latitude"].to_numpy()
    ascending = sampling["ascending"].to_numpy()
    # Between consecutive images, away from the turning points and the dark gaps
    moving = np.abs(np.diff(latitude)) > 0.05
    np.testing.assert_array_equal(
        (np.diff(latitude) > 0)[moving], ascending[:-1][moving]
    )


def test_interval_shorter_than_an_observation_raises():
    with pytest.raises(ValueError, match="shorter than one observation"):
        SHOWSimulator().run(
            ["sampling"],
            {**PERIOD, "acquisition_cfg": {"sampling": {"interval_s": 1.0}}},
        )


def test_show_observation_duration():
    simulator = SHOWSimulator()

    # 22 exposures of 2/22 s, each followed by 16.7 ms of dead time
    assert simulator.run(["observation_duration"])[
        "observation_duration"
    ] == pytest.approx(22 * (2.0 / 22 + 0.0167))
    auto = {"timing": {"integration_time_s": "auto"}}
    assert (
        simulator.run(["observation_duration"], {"acquisition_cfg": auto})[
            "observation_duration"
        ]
        is None
    )


def test_ali_observation_duration_matches_the_l1a_frames():
    acquisition_cfg = {"timing": {"exposure": [0.05, 0.2, 0.3, 0.4, 1.0]}}
    data = ALISimulator().run(
        ["l1a", "observation_duration", "instrument_definition"],
        {**FORCED, "acquisition_cfg": acquisition_cfg, "l1a_cfg": {"noise": False}},
    )

    frames = data["l1a"].to_dataset(inherit=False)
    seconds = (frames["time"] - frames["time"][0]) / np.timedelta64(1, "s")
    detector = data["instrument_definition"]["detector"]
    last = frames.isel(time=-1)
    end = (
        float(seconds[-1])
        + int(last["num_averaged_exposures"])
        * (float(last["exposure_time"]) + detector["dead_time_s"])
        + 0.1  # the nominal pattern's filter change
    )
    assert data["observation_duration"] == pytest.approx(end)
    assert ALISimulator().run(["observation_duration"])["observation_duration"] is None


def test_ali_profiles_are_the_l1b_across_track_bins():
    simulator = ALISimulator()
    sampling = simulator.run(["sampling"], PERIOD)["sampling"]
    sample = sampling.isel(time=sampling.sizes["time"] // 2)

    data = simulator.run(
        ["l1b", "calibration_database"],
        {
            "time": pd.Timestamp(sample["time"].to_numpy().item(), tz="UTC"),
            "l1a_cfg": {"noise": False},
        },
        config={"observation_method": "orbit"},
    )

    # L1A to L1B geolocates each bin with aliprocessing's own camera model and sun
    l1b = (
        data["l1b"]
        .isel(observation_set=0)
        .sel(reference_altitude=25e3)
        .mean("wavelength")
    )
    np.testing.assert_allclose(
        sampling["across_track_angle"], l1b["across_track_angle"]
    )
    np.testing.assert_allclose(
        sample["tangent_latitude"], l1b["tangent_latitude"], atol=0.005
    )
    longitude_difference = (
        sample["tangent_longitude"].to_numpy()
        - l1b["tangent_longitude"].to_numpy()
        + 180.0
    ) % 360.0 - 180.0
    np.testing.assert_allclose(longitude_difference, 0.0, atol=0.005)
    np.testing.assert_allclose(
        sample["solar_zenith_angle"], l1b["solar_zenith_angle"], atol=0.02
    )
    np.testing.assert_allclose(
        sample["relative_solar_azimuth_angle"],
        l1b["relative_solar_azimuth_angle"],
        atol=0.05,
    )
