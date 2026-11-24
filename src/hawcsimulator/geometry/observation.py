from __future__ import annotations

import numpy as np
import sasktran2 as sk
from sasktran2.viewinggeo.base import ViewingGeometryContainer
from skretrieval.retrieval.observation import Observation

from hawcsimulator.geometry.ecef import ECEFLimbGeometry
from hawcsimulator.geometry.utc import utc_times


class SimulatedObservationGeometry(Observation):
    def __init__(
        self,
        viewing_geo: ViewingGeometryContainer,
        sample_wavel: np.array | None = None,
        sample_wavenumber: np.array | None = None,
    ):
        self._viewing_geo = viewing_geo
        if sample_wavel is not None:
            self._sample_wavel = sample_wavel
        elif sample_wavenumber is not None:
            self._sample_wavel = 1e7 / sample_wavenumber
        else:
            msg = "Either sample_wavel or sample_wavenumber must be provided"
            raise ValueError(msg)
        self._reference_latitude = float(
            viewing_geo._geometry_ds["tangent_latitude"].mean()
        )
        self._reference_longitude = float(
            viewing_geo._geometry_ds["tangent_longitude"].mean()
        )

    def sk2_geometry(self, **kwargs) -> dict[sk.ViewingGeometry]:
        """
        The "Ideal" viewing geometry for the observation. One viewing ray for every
        line of sight of the instrument

        Returns
        -------
        dict[sk.ViewingGeometry]
            _description_
        """
        return {"measurement": self._viewing_geo}

    def skretrieval_l1(self, **kwargs):
        """
        The L1 data for the observation in the "Core Radiance Format"

        Returns
        -------
        dict[RadianceGridded]
        """
        # Not used for simulation, should never be called
        raise NotImplementedError()

    def sample_wavelengths(self) -> dict[np.array]:
        """
        The sample wavelengths for the observation in [nm]

        Returns
        -------
        dict[np.array]
        """
        return {"measurement": self._sample_wavel}

    def reference_cos_sza(self) -> dict[float]:
        """
        The reference cosine of the solar zenith angle for the observation

        Returns
        -------
        dict[float]
        """
        return {"measurement": self._viewing_geo.recommended_cos_sza()}

    def reference_latitude(self) -> dict[float]:
        """
        The reference latitude for the observation

        Returns
        -------
        dict[float]
        """
        return {"measurement": self._reference_latitude}

    def reference_longitude(self) -> dict[float]:
        """
        The reference longitude for the observation

        Returns
        -------
        dict[float]
        """
        return {"measurement": self._reference_longitude}

    def append_information_to_l1(self, l1, **kwargs) -> None:
        pass
        # l1.data["tangent_altitude"] = (["los"], self._tangent_altitude)
        # l1.data["tangent_latitude"] = (["los"], self._tangent_latitude)
        # l1.data["tangent_longitude"] = (["los"], self._tangent_longitude)
        # l1.data["tangent_cos_sza"] = (["los"], self._tangent_cos_sza)
        # l1.data["tangent_relative_saa"] = (["los"], self._tangent_relative_saa)
        # l1.data["observer_altitude"] = self._observer_altitude


def observation_geometry(observation: SimulatedObservationGeometry) -> dict:
    """
    Geometry of each line of sight of a simulated limb observation, as the front end
    radiance calculation used it.

    Parameters
    ----------
    observation : SimulatedObservationGeometry

    Returns
    -------
    dict
        Arrays along the lines of sight: tangent_altitude [m], tangent_latitude
        [degrees_north], tangent_longitude [degrees_east, -180 to 180),
        observer_latitude, observer_longitude, observer_altitude [m],
        solar_zenith_angle [degrees], relative_solar_azimuth_angle [degrees, -180 to
        180, 0 is forward scatter] and los_azimuth_angle [degrees clockwise from
        north at the tangent point, 0 to 360)
    """
    viewing_geo = observation._viewing_geo
    if not isinstance(viewing_geo, ECEFLimbGeometry):
        msg = (
            "Simulated observations are built from ECEF lines of sight "
            f"(hawcsimulator.geometry.ecef.ECEFLimbGeometry), not {type(viewing_geo).__name__}"
        )
        raise TypeError(msg)
    geo = viewing_geo.geometry_ds
    return {name: geo[name].to_numpy() for name in _GEOMETRY_NAMES}


_GEOMETRY_NAMES = (
    "tangent_altitude",
    "tangent_latitude",
    "tangent_longitude",
    "observer_latitude",
    "observer_longitude",
    "observer_altitude",
    "solar_zenith_angle",
    "relative_solar_azimuth_angle",
    "los_azimuth_angle",
)


def utc_datetime64(time) -> np.datetime64:
    """``time`` as a timezone-naive UTC numpy datetime64[ns]."""
    return utc_times(time)[0]
