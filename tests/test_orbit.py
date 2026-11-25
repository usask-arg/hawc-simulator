from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sasktran2 as sk

from hawcsimulator.geometry.orbit import (
    EARTH_GM,
    NOMINAL_ORBIT,
    CircularOrbit,
    limb_look_vectors,
    tangent_point_geometry,
    time_at_tangent_latitude,
)

# The example HAWCsat orbit (hawcsimulator/geometry/example_orbit.nc in the sasktran2
# database) comes from another tool. It is sun-synchronous at about 450 km, crosses
# the equator northbound at 01:10:17.24 UTC on 2019-08-01 at 4.93 E (01:30 local
# time), and looks behind the spacecraft. These are its tangent points at 20 km
# (across=256).
EXAMPLE_ORBIT = CircularOrbit.sun_synchronous(
    450e3, 1.5, np.datetime64("2019-08-01T01:10:17.24")
)
EXAMPLE_TANGENT_POINTS = [
    # time, latitude, longitude
    ("2019-08-01T01:22:48", 27.4041, -1.9440),  # along=750
    ("2019-08-01T01:54:28", 30.3662, 178.1106),  # along=2650
]


def _latitude(position: np.ndarray) -> float:
    geodetic = sk.WGS84()
    geodetic.from_xyz(position)
    return geodetic.latitude


def _tangent_point(observer: np.ndarray, look: np.ndarray) -> sk.WGS84:
    geodetic = sk.WGS84()
    geodetic.from_tangent_point(observer, look)
    return geodetic


def test_sun_synchronous_inclination_and_period():
    assert NOMINAL_ORBIT.inclination == pytest.approx(97.21, abs=0.01)

    # J2 lengthens the nodal period of a near-polar orbit by a few seconds
    kepler_period = 2 * np.pi * np.sqrt(NOMINAL_ORBIT.semi_major_axis**3 / EARTH_GM)
    assert NOMINAL_ORBIT.nodal_period == pytest.approx(kepler_period, rel=2e-3)
    assert NOMINAL_ORBIT.nodal_period > kepler_period


def test_spacecraft_crosses_the_equator_northbound_at_the_local_time_of_the_node():
    epoch = NOMINAL_ORBIT.ascending_node_time
    days = np.array([0.0, 30.0, 182.0, 365.0])
    # The first node crossing at least these many days after the epoch
    orbits = np.ceil(days * 86400.0 / NOMINAL_ORBIT.nodal_period)
    times = epoch + (orbits * NOMINAL_ORBIT.nodal_period * 1e9).astype(
        "timedelta64[ns]"
    )

    state = NOMINAL_ORBIT.state(times)

    np.testing.assert_allclose(state.position[:, 2], 0.0, atol=1.0)
    assert np.all(state.along_track[:, 2] > 0)
    longitude = np.degrees(np.arctan2(state.position[:, 1], state.position[:, 0]))
    hours = (times - times.astype("datetime64[D]")) / np.timedelta64(1, "h")
    local_time = (hours + longitude / 15.0) % 24.0
    np.testing.assert_allclose(local_time, 13.5, atol=1e-6)
    np.testing.assert_allclose(
        NOMINAL_ORBIT.local_time_of_ascending_node(times), 13.5, atol=1e-6
    )


def test_spacecraft_stays_on_the_orbit():
    # One orbit, finely enough sampled to find the highest latitude
    times = NOMINAL_ORBIT.ascending_node_time + np.arange(0, 5700).astype(
        "timedelta64[s]"
    )
    state = NOMINAL_ORBIT.state(times)

    np.testing.assert_allclose(
        np.linalg.norm(state.position, axis=1), NOMINAL_ORBIT.semi_major_axis
    )
    np.testing.assert_allclose(np.linalg.norm(state.along_track, axis=1), 1.0)
    np.testing.assert_allclose(
        np.sum(state.position * state.along_track, axis=1), 0.0, atol=1e-6
    )

    geocentric_latitude = np.degrees(
        np.arcsin(state.position[:, 2] / np.linalg.norm(state.position, axis=1))
    )
    assert geocentric_latitude.max() == pytest.approx(
        180.0 - NOMINAL_ORBIT.inclination, abs=0.05
    )


def test_timezones():
    utc = NOMINAL_ORBIT.state(pd.Timestamp("2026-03-20T12:00:00Z"))
    local = NOMINAL_ORBIT.state(pd.Timestamp("2026-03-20T06:00:00-06:00"))
    naive = NOMINAL_ORBIT.state(np.datetime64("2026-03-20T12:00:00"))

    np.testing.assert_array_equal(utc.position, local.position)
    np.testing.assert_array_equal(utc.position, naive.position)


def test_look_vectors_reach_the_tangent_altitudes_behind_the_spacecraft():
    state = NOMINAL_ORBIT.state(np.datetime64("2026-03-20T12:00:00"))
    observer, along_track = state.position[0], state.along_track[0]
    altitudes = np.arange(-3000.0, 56001.0, 500.0)

    looks = limb_look_vectors(observer, along_track, altitudes, look_azimuth=180.0)

    np.testing.assert_allclose(np.linalg.norm(looks, axis=1), 1.0)
    reached = [_tangent_point(observer, look).altitude for look in looks]
    np.testing.assert_allclose(reached, altitudes, atol=1.0)
    # Behind: the horizontal part of each look is opposite the direction of motion
    up = observer / np.linalg.norm(observer)
    horizontal = looks - np.outer(looks @ up, up)
    horizontal /= np.linalg.norm(horizontal, axis=1, keepdims=True)
    forward = along_track - (along_track @ up) * up
    forward /= np.linalg.norm(forward)
    np.testing.assert_allclose(horizontal @ forward, -1.0, atol=1e-4)


def test_look_azimuth_is_clockwise_looking_down():
    state = NOMINAL_ORBIT.state(np.datetime64("2026-03-20T12:00:00"))
    observer, along_track = state.position[0], state.along_track[0]
    ahead = limb_look_vectors(observer, along_track, [25e3], 0.0)[0]
    right = limb_look_vectors(observer, along_track, [25e3], 90.0)[0]

    # Looking down with the direction of motion ahead, 90 degrees is to the right,
    # so the horizontal parts of right, ahead and up are right-handed
    up = observer / np.linalg.norm(observer)
    ahead = ahead - (ahead @ up) * up
    right = right - (right @ up) * up
    assert np.cross(right, ahead) @ up == pytest.approx(
        np.linalg.norm(right) * np.linalg.norm(ahead), rel=1e-3
    )


@pytest.mark.parametrize("ascending", [True, False])
def test_time_at_tangent_latitude(ascending):
    after = np.datetime64("2026-06-21T00:00:00")

    time = time_at_tangent_latitude(
        NOMINAL_ORBIT, 45.0, after, ascending=ascending, tangent_altitude=25e3
    )

    assert time.tz is not None
    assert pd.Timestamp(after, tz="UTC") <= time
    assert time < pd.Timestamp(after, tz="UTC") + pd.Timedelta(
        seconds=NOMINAL_ORBIT.nodal_period
    )
    latitudes = []
    for t in (time, time + pd.Timedelta(seconds=10)):
        state = NOMINAL_ORBIT.state(t)
        look = limb_look_vectors(state.position[0], state.along_track[0], [25e3])[0]
        latitudes.append(_tangent_point(state.position[0], look).latitude)
    assert latitudes[0] == pytest.approx(45.0, abs=1e-3)
    assert (latitudes[1] > latitudes[0]) == ascending


def test_time_at_unreachable_tangent_latitude_raises():
    with pytest.raises(ValueError, match="does not cross"):
        time_at_tangent_latitude(NOMINAL_ORBIT, 89.0, np.datetime64("2026-06-21"))


@pytest.mark.parametrize(("time", "latitude", "longitude"), EXAMPLE_TANGENT_POINTS)
def test_agrees_with_the_example_orbit(time, latitude, longitude):
    # The example orbit is slightly eccentric and this one circular, so they agree
    # to tens of km; looking ahead instead of behind would be thousands of km off
    state = EXAMPLE_ORBIT.state(np.datetime64(time))
    look = limb_look_vectors(state.position[0], state.along_track[0], [20e3], 180.0)[0]
    tangent = _tangent_point(state.position[0], look)

    expected = sk.WGS84()
    expected.from_lat_lon_alt(latitude, longitude, 20e3)
    distance = np.linalg.norm(
        np.asarray(tangent.location) - np.asarray(expected.location)
    )
    assert distance < 50e3


def test_invalid_orbits():
    with pytest.raises(ValueError, match="altitude"):
        CircularOrbit(-1.0, 97.0, np.datetime64("2026-01-01"), 0.0)
    with pytest.raises(ValueError, match="inclination"):
        CircularOrbit(450e3, 190.0, np.datetime64("2026-01-01"), 0.0)


def test_across_track_columns_straddle_the_boresight():
    time = np.datetime64("2026-03-20T12:00:00")
    boresight = tangent_point_geometry(NOMINAL_ORBIT, time).isel(time=0)
    columns = tangent_point_geometry(
        NOMINAL_ORBIT, time, across_track_angles=[-2.0, 0.0, 2.0]
    ).isel(time=0)

    # The middle column is the boresight's line of sight
    for name in ("tangent_latitude", "tangent_longitude", "los_azimuth_angle"):
        assert float(columns[name][1]) == pytest.approx(
            float(boresight[name]), abs=1e-4
        )

    # The outer columns are 2 degrees either side, about 84 km at the tangent point
    def location(i):
        geodetic = sk.WGS84()
        geodetic.from_lat_lon_alt(
            float(columns["tangent_latitude"][i]),
            float(columns["tangent_longitude"][i]),
            25e3,
        )
        return np.asarray(geodetic.location)

    left = np.linalg.norm(location(0) - location(1))
    right = np.linalg.norm(location(2) - location(1))
    assert left == pytest.approx(right, rel=0.01)
    assert left == pytest.approx(2400e3 * np.tan(np.deg2rad(2.0)), rel=0.05)
