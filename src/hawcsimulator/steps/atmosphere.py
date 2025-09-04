from __future__ import annotations

import numpy as np
import sasktran2 as sk
from hamilton.function_modifiers import config
from sasktran2.optical.base import OpticalProperty

import hawcsimulator.atmosphere.earthcare as earthcare
import hawcsimulator.atmosphere.ompscalera as ompscalera
from hawcsimulator.datastructures.atmosphere import Atmosphere
from hawcsimulator.datastructures.viewinggeo import ObservationContainer


@config.when(atmosphere_method="default")
def atmosphere__default(constituents: dict | None = None) -> Atmosphere:
    """
    Constructs an atmosphere manually from a dictionary of sasktran2.Constituent objects.

    Parameters
    ----------
    constituents : dict | None, optional
        A dictionary of sasktran2.Constituent objects, with {key: Constituent}, by default None

    Returns
    -------
    Atmosphere
    """
    if constituents is None:
        constituents = {}
    atmosphere = {
        "rayleigh": sk.constituent.Rayleigh(),
        "o3": sk.climatology.mipas.constituent(
            "O3", sk.optical.O3DBM(), climatology="std"
        ),
        "solar_irradiance": sk.constituent.SolarIrradiance(mode="average"),
        "albedo": sk.constituent.LambertianSurface(0.3),
    }

    for k, v in constituents.items():
        atmosphere[k] = v

    return Atmosphere(atmosphere)


@config.when(atmosphere_method="earthcare")
def atmosphere__earthcare(constituents: dict | None = None) -> Atmosphere:
    """
    Constructs an atmosphere using the EarthCARE configuration.

    Returns
    -------
    Atmosphere
    """
    if constituents is None:
        constituents = {}

    h2o = earthcare._water(30)
    alt_grid = np.arange(0, 65001, 1000.0)
    vmr = np.interp(
        alt_grid, h2o["altitude"].to_numpy(), h2o.to_numpy(), right=h2o.to_numpy()[-1]
    )
    atmosphere = {
        "rayleigh": sk.constituent.Rayleigh(),
        "o3": sk.climatology.mipas.constituent(
            "O3", sk.optical.O3DBM(), climatology="std"
        ),
        "solar_irradiance": sk.constituent.SolarIrradiance(mode="average"),
        "h2o": sk.constituent.VMRAltitudeAbsorber(
            sk.optical.HITRANAbsorber("h2o"), alt_grid, vmr
        ),
        "albedo": sk.constituent.LambertianSurface(0.3),
    }

    for k, v in constituents.items():
        atmosphere[k] = v

    return Atmosphere(atmosphere)


@config.when(atmosphere_method="omps_calipso_era5")
def atmosphere__omps_calipso_era5(
    observation: ObservationContainer,
    aerosol_optical_property: OpticalProperty | None = None,
    h2o_optical_property: OpticalProperty | None = None,
    aerosol_kwargs: dict | None = None,
    constituents: dict | None = None,
) -> Atmosphere:
    if constituents is None:
        constituents = {}
    if aerosol_kwargs is None:
        aerosol_kwargs = {}

    lat = observation.observation.reference_latitude()["measurement"]

    ds = ompscalera.load_data(lat).fillna(0.0)
    alt_grid = np.arange(0, 65001, 1000.0)
    vmr = np.interp(
        alt_grid,
        ds["altitude"].to_numpy() * 1000,
        ds["h2o_vmr"].to_numpy(),
        right=ds["h2o_vmr"].to_numpy()[-1],
    )

    ext = (
        np.interp(
            alt_grid,
            ds["altitude"].to_numpy() * 1000,
            ds["extinction"].to_numpy(),
            right=0,
        )
        / 1000
    )

    atmosphere = {
        "rayleigh": sk.constituent.Rayleigh(),
        "o3": sk.climatology.mipas.constituent(
            "O3", sk.optical.O3DBM(), climatology="std"
        ),
        "solar_irradiance": sk.constituent.SolarIrradiance(mode="average"),
        "albedo": sk.constituent.LambertianSurface(0.3),
    }

    if h2o_optical_property is not None:
        atmosphere["h2o"] = sk.constituent.VMRAltitudeAbsorber(
            h2o_optical_property, alt_grid, vmr
        )
    if aerosol_optical_property is not None:
        if "median_radius" in aerosol_optical_property._psize_dist.args():
            aerosol_kwargs["median_radius"] = np.ones_like(alt_grid) * aerosol_kwargs.pop('uniform_median_radius_nm', 80)
        atmosphere["aerosol"] = sk.constituent.ExtinctionScatterer(
            aerosol_optical_property, alt_grid, ext, **aerosol_kwargs
        )

    for k, v in constituents.items():
        atmosphere[k] = v

    return Atmosphere(atmosphere)
