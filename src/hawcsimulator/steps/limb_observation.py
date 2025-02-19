from __future__ import annotations

import numpy as np
import pandas as pd
import sasktran2 as sk
from hamilton.function_modifiers import config

from hawcsimulator.datastructures.viewinggeo import ObservationContainer
from hawcsimulator.geometry.observation import SimulatedObservationGeometry


@config.when(observation_method="limb")
def observation__limb(
    viewing_tangent_altitudes: np.ndarray,
    time: pd.Timestamp,
    tangent_latitude: float,
    tangent_longitude: float,
    observer_altitude: float,
    sample_wavelengths: np.ndarray,
    tangent_solar_zenith_angle: float | None = None,
    tangent_solar_azimuth_angle: float | None = None,
) -> ObservationContainer:
    """
    Creates an idealized limb viewing observation based on a set of viewing tangent altitudes and
    optionally solar angles at the tangent point.  The observation is created assuming the solar angles
    are the same for every tangent altitude.


    Parameters
    ----------
    viewing_tangent_altitudes : np.array
        Tangent altitudes for the observation in [m], assuming no refraction
    time : pd.Timestamp
        Time of the observation.  Primarily used when the solar angles are not specified to calculate
        the sun position
    tangent_latitude : float
        Tangent latitude in [degrees]
    tangent_longitude : float
        Tangent longitude in [degrees]
    observer_altitude : float
        Altitude of the observer in [m]
    sample_wavelengths : np.ndarray
        Observation sample wavelengths for the instrument in [nm]
    tangent_solar_zenith_angle : float | None, optional
        Solar zenith angle in [degrees], by default None indicating it will be calculated from the observation time
    tangent_solar_azimuth_angle : float | None, optional
        Relative solar azimuth angle in [degrees] where 0 degrees is forward scatter, by default None indicating it will be
        calculated from the observation time

    Returns
    -------
    ObservationContainer
    """
    tan_alts = viewing_tangent_altitudes
    obs_time = time

    if (
        tangent_solar_zenith_angle is not None
        and tangent_solar_azimuth_angle is not None
    ):
        # Forced angles
        solar_handler = sk.solar.SolarGeometryHandlerForced(
            tangent_solar_zenith_angle, tangent_solar_azimuth_angle
        )
    else:
        # Time angles
        solar_handler = sk.solar.SolarGeometryHandlerAstropy()

    viewing_geo = sk.viewinggeo.LimbVertical.from_tangent_parameters(
        solar_handler=solar_handler,
        tangent_altitudes=tan_alts,
        tangent_latitude=tangent_latitude,
        tangent_longitude=tangent_longitude,
        time=obs_time,
        observer_altitude=observer_altitude,
        viewing_azimuth=0.0,
    )

    return ObservationContainer(
        SimulatedObservationGeometry(
            viewing_geo=viewing_geo,
            sample_wavel=sample_wavelengths,
        ),
        obs_time,
    )
