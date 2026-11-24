from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sasktran2 as sk

from hawcsimulator.atmosphere.tropopause import thermal_tropopause_altitude
from hawcsimulator.geometry.observation import observation_geometry, utc_datetime64
from hawcsimulator.steps.limb_observation import observation__limb


def test_observation_geometry_with_forced_solar_angles():
    observation = observation__limb(
        viewing_tangent_altitudes=np.arange(10000.0, 40001.0, 1000.0),
        time=pd.Timestamp("2022-01-01T12:00:00Z"),
        tangent_latitude=30.0,
        tangent_longitude=200.0,
        observer_altitude=450000.0,
        sample_wavelengths=np.array([745.0]),
        tangent_solar_zenith_angle=60.0,
        tangent_solar_azimuth_angle=170.0,
    )

    geometry = observation_geometry(observation.observation)

    np.testing.assert_allclose(geometry["solar_zenith_angle"], 60.0, atol=1e-6)
    np.testing.assert_allclose(
        geometry["relative_solar_azimuth_angle"], 170.0, atol=1e-6
    )
    # Longitudes are wrapped to [-180, 180)
    np.testing.assert_allclose(geometry["tangent_longitude"], -160.0, atol=0.5)
    assert np.all(geometry["observer_longitude"] < 180)
    assert len(geometry["tangent_altitude"]) == 31


def test_utc_datetime64():
    assert utc_datetime64(pd.Timestamp("2022-01-01T12:00:00Z")) == np.datetime64(
        "2022-01-01T12:00:00", "ns"
    )
    assert utc_datetime64(pd.Timestamp("2022-01-01T06:00:00-06:00")) == np.datetime64(
        "2022-01-01T12:00:00", "ns"
    )


def test_thermal_tropopause_of_the_us_standard_atmosphere():
    altitude = np.arange(0, 65001, 1000.0)
    geometry = sk.Geometry1D(
        0.5,
        0.0,
        6372e3,
        altitude,
        sk.InterpolationMethod.LinearInterpolation,
        sk.GeometryType.Spherical,
    )
    atmosphere = sk.Atmosphere(geometry, sk.Config(), wavelengths_nm=np.array([745.0]))
    sk.climatology.us76.add_us76_standard_atmosphere(atmosphere)

    # The US76 lapse rate changes from -6.5 to 0 K/km at 11 km
    assert thermal_tropopause_altitude(
        altitude, atmosphere.temperature_k
    ) == pytest.approx(11000.0, abs=500.0)
