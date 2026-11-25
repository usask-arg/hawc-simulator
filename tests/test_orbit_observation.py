from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sasktran2 as sk
from aliprocessing.geometry import sun_direction, tangent_geometry

from hawcsimulator.geometry.ecef import ECEFLimbGeometry
from hawcsimulator.geometry.observation import observation_geometry
from hawcsimulator.geometry.orbit import (
    NOMINAL_ORBIT,
    limb_look_vectors,
    time_at_tangent_latitude,
)
from hawcsimulator.geometry.solar import SunDirectionSolarHandler
from hawcsimulator.simulator import Simulator
from hawcsimulator.steps.limb_observation import observation__orbit

TANGENT_ALTITUDES = np.arange(0.0, 50001.0, 1000.0)


def _geometry(latitude: float, date: str, look_azimuth: float = 180.0):
    time = time_at_tangent_latitude(
        NOMINAL_ORBIT, latitude, np.datetime64(date), look_azimuth=look_azimuth
    )
    state = NOMINAL_ORBIT.state(time)
    looks = limb_look_vectors(
        state.position[0], state.along_track[0], TANGENT_ALTITUDES, look_azimuth
    )
    return ECEFLimbGeometry(state.time[0], state.position[0], looks)


def test_solar_angles_match_sasktran2s_astropy_handler():
    fast = SunDirectionSolarHandler()
    astropy = sk.solar.SolarGeometryHandlerAstropy()
    time = pd.Timestamp("2026-06-21T13:00:00Z")

    for latitude, longitude in [(45.0, 10.0), (-30.0, 200.0), (70.0, -60.0)]:
        zenith, azimuth = fast.target_solar_angles(latitude, longitude, 20e3, time)
        expected_zenith, expected_azimuth = astropy.target_solar_angles(
            latitude, longitude, 20e3, time
        )
        assert zenith == pytest.approx(expected_zenith, abs=0.01)
        assert (azimuth - expected_azimuth + 180) % 360 - 180 == pytest.approx(
            0.0, abs=0.02
        )


def test_rays_are_the_reported_geometry():
    geometry = _geometry(45.0, "2026-06-21")
    ds = geometry.geometry_ds

    rays = geometry.observer_rays
    assert len(rays) == len(TANGENT_ALTITUDES)
    np.testing.assert_allclose(
        [ray._tangent_altitude_m for ray in rays], ds["tangent_altitude"], atol=1e-6
    )
    np.testing.assert_allclose(
        [ray._cos_sza for ray in rays],
        np.cos(np.deg2rad(ds["solar_zenith_angle"])),
        atol=1e-12,
    )
    relative_azimuth = np.rad2deg([ray._relative_azimuth for ray in rays])
    np.testing.assert_allclose(
        (relative_azimuth - ds["relative_solar_azimuth_angle"] + 180) % 360 - 180,
        0.0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        [ray._observer_altitude_m for ray in rays], ds["observer_altitude"]
    )
    np.testing.assert_allclose(ds["tangent_altitude"], TANGENT_ALTITUDES, atol=1.0)


def test_geometry_matches_aliprocessing():
    # aliprocessing's L1A to L1B geolocation is an independent implementation, with
    # its own low-precision sun
    geometry = _geometry(30.0, "2026-03-20")
    ds = geometry.geometry_ds

    expected = tangent_geometry(
        ds["observer_position_ecef"].to_numpy()[0],
        ds["look_direction_ecef"].to_numpy(),
        sun_direction(ds["time"].to_numpy()[0]),
    )

    np.testing.assert_allclose(ds["tangent_latitude"], expected.latitude, atol=1e-6)
    np.testing.assert_allclose(
        ds["tangent_longitude"], (expected.longitude + 180) % 360 - 180, atol=1e-6
    )
    np.testing.assert_allclose(
        ds["solar_zenith_angle"], expected.solar_zenith_angle, atol=0.02
    )
    np.testing.assert_allclose(
        ds["relative_solar_azimuth_angle"],
        expected.relative_solar_azimuth_angle,
        atol=0.02,
    )
    np.testing.assert_allclose(
        ds["los_azimuth_angle"], expected.los_azimuth_angle, atol=1e-6
    )


@pytest.mark.parametrize(
    "date", ["2026-03-20", "2026-06-21", "2026-09-22", "2026-12-21"]
)
def test_looking_behind_puts_forward_scatter_in_the_northern_hemisphere(date):
    north = _geometry(45.0, date).geometry_ds["relative_solar_azimuth_angle"]
    south = _geometry(-45.0, date).geometry_ds["relative_solar_azimuth_angle"]

    assert np.all(np.abs(north) < 60.0)
    assert np.all(np.abs(south) > 120.0)


def test_observation_from_the_simulator():
    time = time_at_tangent_latitude(NOMINAL_ORBIT, 30.0, np.datetime64("2026-06-21"))

    data = Simulator().run(
        ["observation"],
        {
            "viewing_tangent_altitudes": TANGENT_ALTITUDES,
            "time": time,
            "sample_wavelengths": np.array([745.0]),
        },
        config={"observation_method": "orbit"},
    )

    observation = data["observation"]
    assert observation.time == time
    geometry = observation_geometry(observation.observation)
    np.testing.assert_allclose(
        geometry["tangent_altitude"], TANGENT_ALTITUDES, atol=1.0
    )
    reference = np.interp(25e3, TANGENT_ALTITUDES, geometry["tangent_latitude"])
    assert reference == pytest.approx(30.0, abs=1e-3)
    assert np.all(geometry["observer_altitude"] > 450e3)
    assert np.all(geometry["observer_altitude"] < 475e3)
    cos_sza = observation.observation.reference_cos_sza()["measurement"]
    assert np.rad2deg(np.arccos(cos_sza)) == pytest.approx(
        np.interp(25e3, TANGENT_ALTITUDES, geometry["solar_zenith_angle"])
    )


def test_night_side_observation_raises():
    time = time_at_tangent_latitude(
        NOMINAL_ORBIT, 0.0, np.datetime64("2026-06-21"), ascending=False
    )

    with pytest.raises(ValueError, match="SZA"):
        observation__orbit(
            viewing_tangent_altitudes=TANGENT_ALTITUDES,
            time=time,
            sample_wavelengths=np.array([745.0]),
        )


@pytest.mark.parametrize(
    ("latitude", "date", "forward"),
    [(75.0, "2026-06-21", True), (-75.0, "2026-12-21", False)],
)
def test_summer_pole_is_daylit_southbound_too(latitude, date, forward):
    # Past the most northerly or southerly point the tangent point stays in the
    # midnight sun, and the scattering still depends on the hemisphere
    time = time_at_tangent_latitude(
        NOMINAL_ORBIT, latitude, np.datetime64(date), ascending=False
    )

    observation = observation__orbit(
        viewing_tangent_altitudes=TANGENT_ALTITUDES,
        time=time,
        sample_wavelengths=np.array([745.0]),
    )

    geometry = observation_geometry(observation.observation)
    relative_azimuth = np.abs(geometry["relative_solar_azimuth_angle"])
    if forward:
        assert np.all(relative_azimuth < 60.0)
    else:
        assert np.all(relative_azimuth > 120.0)
