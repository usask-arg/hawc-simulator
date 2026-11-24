from __future__ import annotations

import numpy as np
import sasktran2 as sk
import xarray as xr

from hawcsimulator.datastructures.viewinggeo import ObservationContainer
from hawcsimulator.geometry.observation import observation_geometry, utc_datetime64

# Tangent altitude whose line of sight gives the truth's location [m]
_REFERENCE_ALTITUDE = 25000.0


def truth(
    sk2_atmosphere: sk.Atmosphere,
    program_of_record: xr.Dataset,
    observation: ObservationContainer,
) -> xr.Dataset:
    """
    The true atmospheric state the front end radiance was calculated from, on the
    simulation's altitude grid.

    Returns
    -------
    xr.Dataset
        On ``altitude`` [m]: pressure [Pa] and temperature [K]; ``<name>_vmr`` for
        each gas (e.g. h2o_vmr, o3_vmr); for each scatterer given by extinction (e.g.
        "aerosol"), ``<name>_extinction`` [m-1] at the wavelength in its
        ``wavelength_nm`` attribute and its particle size parameters (e.g.
        aerosol_median_radius [nm]); and surface_albedo and tropopause_altitude [m]
        as scalars. Attributes give the time and the tangent location of the line of
        sight nearest 25 km. ``truth.to_netcdf(path)`` saves it.
    """
    altitude = np.asarray(sk2_atmosphere.model_geometry.altitudes(), dtype=float)
    ds = xr.Dataset(
        {
            "pressure": ("altitude", np.asarray(sk2_atmosphere.pressure_pa)),
            "temperature": ("altitude", np.asarray(sk2_atmosphere.temperature_k)),
        },
        coords={"altitude": altitude},
    )
    ds["pressure"].attrs["units"] = "Pa"
    ds["temperature"].attrs["units"] = "K"
    ds["altitude"].attrs["units"] = "m"

    # The constituents the radiance was calculated with, including any the front end
    # step added to the user's atmosphere
    for name, constituent in sk2_atmosphere._constituents.items():
        if isinstance(constituent, sk.constituent.VMRAltitudeAbsorber):
            ds[f"{name}_vmr"] = _on_grid(
                altitude, constituent.altitudes_m, constituent.vmr, "1"
            )
        elif isinstance(constituent, sk.constituent.ExtinctionScatterer):
            extinction = _on_grid(
                altitude, constituent.altitudes_m, constituent.extinction_per_m, "m-1"
            )
            extinction.attrs["wavelength_nm"] = float(
                constituent._extinction_wavelength_nm
            )
            ds[f"{name}_extinction"] = extinction
            for parameter, raw in constituent._kwargs.items():
                values = np.asarray(raw, dtype=float)
                if values.shape == np.shape(constituent.altitudes_m):
                    ds[f"{name}_{parameter}"] = _on_grid(
                        altitude,
                        constituent.altitudes_m,
                        values,
                        "nm" if parameter.endswith("radius") else "1",
                    )
        elif isinstance(constituent, sk.constituent.LambertianSurface):
            albedo = np.atleast_1d(np.asarray(constituent.albedo, dtype=float))
            if albedo.size == 1:
                ds["surface_albedo"] = ((), float(albedo[0]), {"units": "1"})

    ds["tropopause_altitude"] = (
        (),
        float(program_of_record["tropopause_altitude"].isel(time=0)),
        {"units": "m"},
    )
    geometry = observation_geometry(observation.observation)
    i = int(np.argmin(np.abs(geometry["tangent_altitude"] - _REFERENCE_ALTITUDE)))
    ds.attrs.update(
        {
            "time": str(utc_datetime64(observation.time)),
            "tangent_latitude": float(geometry["tangent_latitude"][i]),
            "tangent_longitude": float(geometry["tangent_longitude"][i]),
        }
    )
    return ds


def _on_grid(altitude: np.ndarray, from_altitude, values, units: str) -> xr.DataArray:
    """``values`` on ``altitude``: held at the bottom, zero above the top."""
    return xr.DataArray(
        np.interp(
            altitude,
            np.asarray(from_altitude, dtype=float),
            np.asarray(values, dtype=float),
            right=0.0,
        ),
        dims="altitude",
        attrs={"units": units},
    )


def on_l2_grid(
    truth: xr.Dataset, names: dict[str, str], like: xr.DataArray, wavelength_nm: float
) -> xr.Dataset:
    """
    Truth variables on an L2 product's grid: interpolated to ``like``'s altitudes,
    renamed with ``names`` (truth name to L2 name) and broadcast over ``like``'s other
    dimensions. Extinction is left out unless the truth's is at ``wavelength_nm``,
    the L2's.
    """
    altitude = like["altitude"]
    profile_like = like.isel(altitude=0, drop=True)
    out = {}
    for truth_name, l2_name in names.items():
        if truth_name not in truth:
            continue
        variable = truth[truth_name]
        if truth_name.endswith("_extinction") and not np.isclose(
            variable.attrs["wavelength_nm"], wavelength_nm
        ):
            continue
        if "altitude" in variable.dims:
            variable = variable.interp(altitude=altitude.to_numpy()).assign_coords(
                altitude=altitude
            )
            out[l2_name] = variable.broadcast_like(like)
        else:
            out[l2_name] = variable.broadcast_like(profile_like)
    result = xr.Dataset(out)
    result.attrs["description"] = "True atmospheric state on the L2 grid"
    return result
