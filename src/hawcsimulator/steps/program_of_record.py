from __future__ import annotations

import numpy as np
import sasktran2 as sk
import xarray as xr

from hawcsimulator.atmosphere.tropopause import thermal_tropopause_altitude
from hawcsimulator.datastructures.atmosphere import Atmosphere
from hawcsimulator.datastructures.viewinggeo import ObservationContainer
from hawcsimulator.geometry.observation import utc_datetime64


def program_of_record(
    sk2_atmosphere: sk.Atmosphere,
    observation: ObservationContainer,
    atmosphere: Atmosphere,
    por_aerosol_wavelength_nm: float | None = None,
) -> xr.Dataset:
    """
    The program of record (POR) the L2 retrievals use for pressure and temperature,
    taken from the simulation atmosphere.

    Parameters
    ----------
    sk2_atmosphere : sk.Atmosphere
        The atmosphere used to calculate the front end radiance
    observation : ObservationContainer
        The observation, for its time
    atmosphere : Atmosphere
        The simulation atmosphere, for its aerosol
    por_aerosol_wavelength_nm : float | None, optional
        When set and the simulation atmosphere has an "aerosol", the POR also holds
        its extinction [m-1] at this wavelength (the truth) as aerosol_extinction.
        SHOW's retrieval then starts from it and uses it as its aerosol prior. By
        default None, which leaves it out

    Returns
    -------
    xr.Dataset
        pressure [Pa], temperature [K] and tropopause_altitude [m] on altitude [m],
        and aerosol_extinction [m-1] when requested, along a time dimension of length
        one
    """
    altitude = sk2_atmosphere.model_geometry.altitudes()
    por = xr.Dataset(
        {
            "temperature": (["altitude"], sk2_atmosphere.temperature_k),
            "pressure": (["altitude"], sk2_atmosphere.pressure_pa),
            "tropopause_altitude": thermal_tropopause_altitude(
                altitude, sk2_atmosphere.temperature_k
            ),
        },
        coords={"altitude": altitude},
    )
    aerosol = atmosphere.constituents.get("aerosol")
    if por_aerosol_wavelength_nm is not None and aerosol is not None:
        por["aerosol_extinction"] = (
            ["altitude"],
            _aerosol_extinction(aerosol, altitude, por_aerosol_wavelength_nm),
        )

    por = xr.concat([por], dim="time")
    por.coords["time"] = [utc_datetime64(observation.time)]

    return por


def _aerosol_extinction(
    aerosol: sk.constituent.ExtinctionScatterer,
    altitude: np.ndarray,
    wavelength_nm: float,
) -> np.ndarray:
    """The aerosol's extinction [m-1] on ``altitude`` at ``wavelength_nm``."""
    reference = getattr(aerosol, "_extinction_wavelength_nm", None)
    if reference is None or not np.isclose(reference, wavelength_nm):
        msg = (
            f"The atmosphere's aerosol extinction is given at {reference} nm, not the "
            f"{wavelength_nm} nm the program of record needs; set "
            "por_aerosol_wavelength_nm to None to leave the POR aerosol out"
        )
        raise ValueError(msg)
    return np.interp(
        altitude,
        np.asarray(aerosol.altitudes_m),
        np.asarray(aerosol.extinction_per_m),
        right=0.0,
    )
