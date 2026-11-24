from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hawcsimulator.geometry.observation import observation_geometry
from hawcsimulator.steps.limb_observation import observation__limb

# Samples from the example HAWCsat orbit (hawcsimulator/geometry/example_orbit.nc, across=256)
DAYSIDE = {  # along=2650, used by the quickstart
    "tangent_latitude": 30.3662,
    "tangent_longitude": 178.1106,
    "observer_latitude": 9.9504,
    "observer_longitude": 175.1511,
    "time": pd.Timestamp("2019-08-01T01:54:28"),
}
NIGHTSIDE = {  # along=750
    "tangent_latitude": 27.4041,
    "tangent_longitude": -1.944,
    "observer_latitude": 47.8036,
    "observer_longitude": -6.189,
    "time": pd.Timestamp("2019-08-01T01:22:48"),
}


def _observation(**kwargs):
    return observation__limb(
        viewing_tangent_altitudes=np.arange(10000.0, 40001.0, 1000.0),
        observer_altitude=450000.0,
        sample_wavelengths=np.array([745.0]),
        **kwargs,
    )


def _solar_zenith_angle(observation) -> float:
    cos_sza = observation.observation.reference_cos_sza()["measurement"]
    return float(np.rad2deg(np.arccos(cos_sza)))


def test_forced_solar_angles():
    observation = _observation(
        tangent_latitude=30.0,
        tangent_longitude=0.0,
        time=pd.Timestamp("2022-01-01T12:00:00Z"),
        tangent_solar_zenith_angle=60.0,
        tangent_solar_azimuth_angle=0.0,
    )

    assert _solar_zenith_angle(observation) == pytest.approx(60.0, abs=0.5)


def test_time_based_solar_angles():
    observation = _observation(**DAYSIDE)

    assert 20.0 < _solar_zenith_angle(observation) < 32.0


def test_observer_location_or_look_azimuth():
    # The observer is placed behind the tangent point along los_azimuth when its
    # location is not given
    without_observer = {
        k: v for k, v in DAYSIDE.items() if not k.startswith("observer_")
    }
    geometry = observation_geometry(
        _observation(**without_observer, los_azimuth=90.0).observation
    )
    reference = int(np.argmin(np.abs(geometry["tangent_altitude"] - 25000.0)))
    assert geometry["los_azimuth_angle"][reference] == pytest.approx(90.0, abs=0.01)
    np.testing.assert_allclose(geometry["observer_altitude"], 450000.0, atol=1.0)
    assert geometry["observer_longitude"][0] < DAYSIDE["tangent_longitude"]

    with pytest.raises(ValueError, match="observer_latitude and observer_longitude"):
        _observation(**{**without_observer, "observer_latitude": 10.0})


def test_lines_of_sight_reach_their_tangent_points():
    observation = _observation(**DAYSIDE)
    geometry = observation_geometry(observation.observation)

    np.testing.assert_allclose(
        geometry["tangent_altitude"], np.arange(10000.0, 40001.0, 1000.0), atol=1.0
    )
    # Every line of sight is in one vertical plane from one observer, near the
    # given observer location
    assert np.ptp(geometry["observer_latitude"]) == 0
    assert geometry["observer_latitude"][0] == pytest.approx(
        DAYSIDE["observer_latitude"], abs=0.5
    )
    assert np.ptp(geometry["los_azimuth_angle"]) < 0.5
    # The reference tangent point is the requested one
    reference = int(np.argmin(np.abs(geometry["tangent_altitude"] - 25000.0)))
    assert geometry["tangent_latitude"][reference] == pytest.approx(
        DAYSIDE["tangent_latitude"], abs=1e-3
    )
    assert geometry["tangent_longitude"][reference] == pytest.approx(
        DAYSIDE["tangent_longitude"], abs=1e-3
    )


def test_forced_angles_are_at_every_tangent_point():
    observation = _observation(
        tangent_latitude=30.0,
        tangent_longitude=0.0,
        time=pd.Timestamp("2022-01-01T12:00:00Z"),
        tangent_solar_zenith_angle=60.0,
        tangent_solar_azimuth_angle=120.0,
    )
    geometry = observation_geometry(observation.observation)

    np.testing.assert_allclose(geometry["solar_zenith_angle"], 60.0)
    np.testing.assert_allclose(geometry["relative_solar_azimuth_angle"], 120.0)


def test_nightside_observation_is_rejected():
    with pytest.raises(ValueError, match="greater than the allowed maximum"):
        _observation(**NIGHTSIDE)
