from __future__ import annotations

import numpy as np
import pandas as pd
import sasktran2 as sk
from hamilton.function_modifiers import config

from hawcsimulator.datastructures.viewinggeo import ObservationContainer
from hawcsimulator.geometry.ecef import ECEFLimbGeometry
from hawcsimulator.geometry.observation import SimulatedObservationGeometry

# Tangent altitude that places the observation and gives its model geometry [m]
REFERENCE_ALTITUDE = 25000.0


@config.when(observation_method="limb")
def observation__limb(
    viewing_tangent_altitudes: np.ndarray,
    time: pd.Timestamp,
    tangent_latitude: float,
    tangent_longitude: float,
    observer_altitude: float,
    sample_wavelengths: np.ndarray,
    observer_longitude: float | None = None,
    observer_latitude: float | None = None,
    los_azimuth: float = 0.0,
    tangent_solar_zenith_angle: float | None = None,
    tangent_solar_azimuth_angle: float | None = None,
    maximum_allowed_sza: float = 88,
) -> ObservationContainer:
    """
    A limb observation from one observer: a vertical scan with one line of sight per
    viewing tangent altitude, each an ECEF ray
    (hawcsimulator.geometry.ecef.ECEFLimbGeometry).

    The lines of sight lie in the vertical plane through the tangent point at
    ``tangent_latitude``, ``tangent_longitude`` and 25 km, looking along
    ``los_azimuth`` there, or from ``observer_latitude`` and ``observer_longitude``
    when they are given. The observer is where that line of sight is at
    ``observer_altitude``, so it is only near a given observer location.

    Parameters
    ----------
    viewing_tangent_altitudes : np.array
        Tangent altitudes for the observation in [m], assuming no refraction
    time : pd.Timestamp
        Time of the observation, which gives the sun's position when the solar angles
        are not forced. Times without a timezone are UTC
    tangent_latitude : float
        Tangent latitude in [degrees]
    tangent_longitude : float
        Tangent longitude in [degrees]
    observer_altitude : float
        Altitude of the observer in [m]
    sample_wavelengths : np.ndarray
        Observation sample wavelengths for the instrument in [nm]
    observer_latitude, observer_longitude : float | None, optional
        Approximate location of the observer in [degrees], which sets the direction
        of the lines of sight. Give both or neither, by default None
    los_azimuth : float, optional
        Azimuth of the lines of sight at the tangent point in [degrees] clockwise from
        north, used when the observer's location is not given, by default 0
    tangent_solar_zenith_angle : float | None, optional
        Solar zenith angle in [degrees] at every tangent point, by default None
        indicating the sun's position at ``time``
    tangent_solar_azimuth_angle : float | None, optional
        Relative solar azimuth angle in [degrees] at every tangent point, where 0
        degrees is forward scatter, by default None indicating the sun's position at
        ``time``
    maximum_allowed_sza : float, optional
        Largest solar zenith angle at 25 km in [degrees], by default 88

    Returns
    -------
    ObservationContainer
    """
    forced = (
        tangent_solar_zenith_angle is not None
        and tangent_solar_azimuth_angle is not None
    )
    if (observer_latitude is None) != (observer_longitude is None):
        msg = "Give both observer_latitude and observer_longitude, or neither"
        raise ValueError(msg)

    geodetic = sk.WGS84()
    if observer_latitude is not None:
        # The observer's location sets the direction the lines of sight look in
        geodetic.from_lat_lon_alt(
            observer_latitude, observer_longitude, observer_altitude
        )
        given_observer = np.asarray(geodetic.location)
    geodetic.from_lat_lon_alt(tangent_latitude, tangent_longitude, REFERENCE_ALTITUDE)
    tangent_point = np.asarray(geodetic.location)
    north = -np.asarray(geodetic.local_south)
    east = -np.asarray(geodetic.local_west)
    if observer_latitude is not None:
        towards = tangent_point - given_observer
        los_azimuth = float(np.degrees(np.arctan2(towards @ east, towards @ north)))
    azimuth = np.deg2rad(los_azimuth)
    horizontal = np.cos(azimuth) * north + np.sin(azimuth) * east
    # The observer is where the line of sight through the tangent point enters the
    # observer's altitude
    observer = np.asarray(
        geodetic.altitude_intercepts(observer_altitude, tangent_point, horizontal)[0]
    )
    looks = np.array(
        [
            geodetic.from_tangent_altitude(altitude, observer, horizontal)
            for altitude in np.atleast_1d(viewing_tangent_altitudes)
        ]
    )

    viewing_geo = ECEFLimbGeometry(
        time,
        observer,
        looks,
        forced_solar_angles=(
            (tangent_solar_zenith_angle, tangent_solar_azimuth_angle)
            if forced
            else None
        ),
        reference_altitude=REFERENCE_ALTITUDE,
    )

    obs_sza = np.rad2deg(np.arccos(viewing_geo.recommended_cos_sza()))
    if obs_sza > maximum_allowed_sza:
        msg = f"Observation SZA: {obs_sza} is greater than the allowed maximum: {maximum_allowed_sza}"
        raise ValueError(msg)

    return ObservationContainer(
        SimulatedObservationGeometry(
            viewing_geo=viewing_geo,
            sample_wavel=sample_wavelengths,
        ),
        time,
    )
