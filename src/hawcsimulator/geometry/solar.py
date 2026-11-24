from __future__ import annotations

import numpy as np
import sasktran2 as sk
from astropy.coordinates import ITRS, get_sun
from astropy.time import Time

from hawcsimulator.geometry.utc import utc_times


class SunDirectionSolarHandler(sk.solar.SolarGeometryHandlerBase):
    """
    Solar angles from the direction of the sun seen from the Earth's centre.

    astropy calculates the sun's Earth-fixed direction once for each time, and each
    location's angles follow from its local axes. That makes many locations at
    one time far cheaper than sasktran2's SolarGeometryHandlerAstropy, which
    transforms the sun to every location separately. Parallax (under 0.003
    degrees) is ignored.
    """

    def __init__(self) -> None:
        super().__init__()
        self._sun: dict[np.datetime64, np.ndarray] = {}

    def sun_direction(self, time) -> np.ndarray:
        """Unit vector from the Earth's centre towards the sun, in ECEF."""
        key = utc_times(time)[0]
        if key not in self._sun:
            t = Time(key, scale="utc")
            sun = get_sun(t).transform_to(ITRS(obstime=t)).cartesian.xyz.to_value()
            self._sun[key] = sun / np.linalg.norm(sun)
        return self._sun[key]

    def target_solar_angles(
        self, latitude: float, longitude: float, altitude: float, time
    ) -> tuple[float, float]:
        """
        Solar zenith angle and solar azimuth (clockwise from north, towards the sun)
        in degrees.
        """
        sun = self.sun_direction(time)
        geodetic = sk.WGS84()
        geodetic.from_lat_lon_alt(latitude, longitude, altitude)
        up = np.asarray(geodetic.local_up)
        north = -np.asarray(geodetic.local_south)
        east = -np.asarray(geodetic.local_west)

        zenith = np.degrees(np.arccos(np.clip(up @ sun, -1.0, 1.0)))
        azimuth = np.degrees(np.arctan2(east @ sun, north @ sun)) % 360.0
        return float(zenith), float(azimuth)
