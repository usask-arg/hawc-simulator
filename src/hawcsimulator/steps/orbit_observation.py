from __future__ import annotations
from typing import Tuple
from pathlib import Path, WindowsPath, PosixPath

import numpy as np
import pandas as pd
import sasktran2 as sk
from hamilton.function_modifiers import config

from hawcsimulator.datastructures.viewinggeo import ObservationContainer
from hawcsimulator.geometry.observation import SimulatedObservationGeometry
from skplatform.scripting.interface import simulate_orbit, open_tle
from skplatform.scripting.instruments import SimulatorInstrument
  

@config.when(observation_method="orbit")
def observation__orbit(
    viewing_tangent_altitudes: np.ndarray,
    time: pd.Timestamp,
    sample_wavelengths: np.ndarray,
    tle: str,  
    boresight_angle_deg: float,
    target_tangent_altitude_m: float,
    # column: int = 0,  # unused, will support selecting the vertical column or be an offset angle 
) -> ObservationContainer:
    """
    Creates an idealized limb viewing observation based on an orbit simulation and a set of viewing 
    tangent altitudes. Solar angles are caclulated from the satellite position. The observation is 
    created assuming the solar angles are the same for every tangent altitude.


    Parameters
    ----------
    viewing_tangent_altitudes : np.array
        Tangent altitudes for the observation in [m], assuming no refraction
    time : pd.Timestamp
        Time of the observation.  The orbit will be propagated to this time. 
    sample_wavelengths : np.ndarray
        Observation sample wavelengths for the instrument in [nm]
    tle : str
        A string with the path and filename of a Two Line Element (TLE) set.  The TLE defines the 
        orbital elements of the orbit to be propagated.  TLEs are only considered valid for a 
        couple of weeks from their epoch, which is the time and date the orbital elements are for, so care 
        must be taken to use a TLE valid for the time requested. 
    boresight_angle_deg : float
        The angle, in [degrees], from the satellite's velocity that the satellite instrument is looking in.  
        The angle is in the plane perpendicular to the up vector of the satellite.  An angle of 180.0 degrees
        is an instrument looking backwards.  The range is -180.0 to 360.0 degrees.
    target_tangent_altitude_m : float
        The tangent altitude, in [m], that the satellite instrument is looking at. This sets the 
        altitude at which the solar angles are calculated and is independent of the viewing_tangent_altitudes. 

    Returns
    -------
    ObservationContainer
    """
    
    tan_alts = viewing_tangent_altitudes
    obs_time = time
    
    tle = open_tle(tle)
    sim_inst = SimulatorInstrument(tle=tle, look_angle_deg=boresight_angle_deg, 
                                   target_altitude_m=target_tangent_altitude_m)
    # length is short because only a single observation is needed
    ds = simulate_orbit(sim_inst, start=time, length=0.001)   
    
    solar_handler = sk.solar.SolarGeometryHandlerAstropy()
    viewing_geo = sk.viewinggeo.LimbVertical.from_tangent_parameters(
        solar_handler=solar_handler,
        tangent_altitudes=tan_alts,
        tangent_latitude=ds['latitude'].item(),
        tangent_longitude=ds['longitude'].item(),
        reference_altitude=ds['altitude'].item(),
        time=obs_time,
        observer_altitude=ds['observer_altitude'].item(),
        viewing_azimuth=ds['azimuth_angle'].item(),
    )

    return ObservationContainer(
        SimulatedObservationGeometry(
            viewing_geo=viewing_geo,
            sample_wavel=sample_wavelengths,
        ),
        obs_time,
    )
