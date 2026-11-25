"""
Spacecraft positions along a circular orbit, and the limb lines of sight from them.

The orbit is propagated analytically: a circular orbit whose node and argument of
latitude drift at the secular rates the Earth's oblateness (J2) causes. That
places observations realistically in latitude, season and local solar time; it
is not an orbit determination.

Everything is in Earth-fixed (ECEF) coordinates. The Earth rotates at the
sidereal rate about the z axis; precession, nutation and polar motion are
ignored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import sasktran2 as sk
import xarray as xr
from scipy.optimize import brentq

from hawcsimulator.geometry.solar import sun_directions
from hawcsimulator.geometry.utc import utc_times

EARTH_GM = 3.986004418e14  # [m^3 s^-2]
EARTH_EQUATORIAL_RADIUS = 6378137.0  # [m], WGS84
EARTH_J2 = 1.08262668e-3
# The Earth's rotation relative to the equinox, and of the mean sun relative to the
# equinox: their difference is one turn per (mean solar) day
SIDEREAL_RATE = np.deg2rad(360.98564736629) / 86400.0  # [rad s^-1]
MEAN_SUN_RATE = SIDEREAL_RATE - 2.0 * np.pi / 86400.0  # [rad s^-1]


@dataclass(frozen=True)
class OrbitState:
    """
    Spacecraft positions along an orbit.

    Attributes
    ----------
    time : np.ndarray
        Times, (time,), UTC datetime64[ns]
    position : np.ndarray
        Spacecraft positions, (time, 3), ECEF [m]
    along_track : np.ndarray
        The direction the spacecraft moves through inertial space, (time, 3), unit
        vectors in ECEF axes. It lies in the orbital plane. The motion over the
        ground differs from it by the Earth's rotation, by up to about 3.5 degrees
        in azimuth at the equator.
    """

    time: np.ndarray
    position: np.ndarray
    along_track: np.ndarray


@dataclass(frozen=True)
class CircularOrbit:
    """
    A circular orbit, with the secular drift of its node and argument of latitude
    from J2.

    Parameters
    ----------
    altitude : float
        Altitude above the equatorial radius [m]. The orbit's semi-major axis is
        the equatorial radius plus this.
    inclination : float
        Inclination [degrees]
    ascending_node_time
        A time the spacecraft crosses the ascending node, the orbit's epoch. Times
        without a timezone are UTC.
    ascending_node_longitude : float
        Longitude of the ascending node at ``ascending_node_time`` [degrees east]
    """

    altitude: float
    inclination: float
    ascending_node_time: np.datetime64
    ascending_node_longitude: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "ascending_node_time", utc_times(self.ascending_node_time)[0]
        )
        if not self.altitude > 0:
            msg = f"altitude must be positive, got {self.altitude}"
            raise ValueError(msg)
        if not 0.0 <= self.inclination <= 180.0:
            msg = (
                f"inclination must be between 0 and 180 degrees, got {self.inclination}"
            )
            raise ValueError(msg)

    @classmethod
    def sun_synchronous(
        cls,
        altitude: float,
        local_time_of_ascending_node: float,
        ascending_node_time,
    ) -> CircularOrbit:
        """
        A sun-synchronous orbit: its node precesses with the mean sun, so it crosses
        the equator northbound at the same local mean solar time every orbit.

        Parameters
        ----------
        altitude : float
            Altitude above the equatorial radius [m]
        local_time_of_ascending_node : float
            Local mean solar time of the ascending node [hours], e.g. 13.5 for 13:30
        ascending_node_time
            A time the spacecraft crosses the ascending node
        """
        semi_major_axis = EARTH_EQUATORIAL_RADIUS + altitude
        mean_motion = np.sqrt(EARTH_GM / semi_major_axis**3)
        cos_inclination = -MEAN_SUN_RATE / (
            1.5
            * EARTH_J2
            * (EARTH_EQUATORIAL_RADIUS / semi_major_axis) ** 2
            * mean_motion
        )
        if abs(cos_inclination) > 1:
            msg = f"There is no sun-synchronous circular orbit at {altitude} m"
            raise ValueError(msg)

        epoch = utc_times(ascending_node_time)[0]
        node_longitude = 15.0 * (local_time_of_ascending_node - _utc_hours(epoch))
        return cls(
            altitude=altitude,
            inclination=float(np.degrees(np.arccos(cos_inclination))),
            ascending_node_time=epoch,
            ascending_node_longitude=float((node_longitude + 180.0) % 360.0 - 180.0),
        )

    @property
    def semi_major_axis(self) -> float:
        """[m]"""
        return EARTH_EQUATORIAL_RADIUS + self.altitude

    @property
    def nodal_precession_rate(self) -> float:
        """Rate the ascending node moves relative to the equinox [rad s^-1]."""
        return -self._j2_factor * self._mean_motion * np.cos(self._inclination_rad)

    @property
    def argument_of_latitude_rate(self) -> float:
        """Rate the spacecraft moves around the orbit from the node [rad s^-1]."""
        cos2 = np.cos(self._inclination_rad) ** 2
        return self._mean_motion * (1.0 + self._j2_factor * (4.0 * cos2 - 1.0))

    @property
    def nodal_period(self) -> float:
        """Time between ascending node crossings [s]."""
        return 2.0 * np.pi / self.argument_of_latitude_rate

    def local_time_of_ascending_node(self, time) -> np.ndarray:
        """Local mean solar time of the ascending node at ``time`` [hours]."""
        times = utc_times(time)
        node_longitude = np.degrees(self._node_longitude(times))
        return (_utc_hours(times) + node_longitude / 15.0) % 24.0

    def state(self, time) -> OrbitState:
        """Spacecraft positions and directions of motion at ``time``."""
        times = utc_times(time)
        elapsed = self._elapsed(times)
        node = self._node_longitude(times)
        argument_of_latitude = self.argument_of_latitude_rate * elapsed
        inclination = self._inclination_rad

        # In-plane unit vectors towards the ascending node, and 90 degrees past it
        towards_node = np.stack(
            [np.cos(node), np.sin(node), np.zeros_like(node)], axis=-1
        )
        past_node = np.stack(
            [
                -np.sin(node) * np.cos(inclination),
                np.cos(node) * np.cos(inclination),
                np.full_like(node, np.sin(inclination)),
            ],
            axis=-1,
        )
        cos_u = np.cos(argument_of_latitude)[:, np.newaxis]
        sin_u = np.sin(argument_of_latitude)[:, np.newaxis]
        return OrbitState(
            time=times,
            position=self.semi_major_axis * (cos_u * towards_node + sin_u * past_node),
            along_track=cos_u * past_node - sin_u * towards_node,
        )

    @property
    def _mean_motion(self) -> float:
        return np.sqrt(EARTH_GM / self.semi_major_axis**3)

    @property
    def _j2_factor(self) -> float:
        return 1.5 * EARTH_J2 * (EARTH_EQUATORIAL_RADIUS / self.semi_major_axis) ** 2

    @property
    def _inclination_rad(self) -> float:
        return np.deg2rad(self.inclination)

    def _elapsed(self, times: np.ndarray) -> np.ndarray:
        return (times - self.ascending_node_time) / np.timedelta64(1, "s")

    def _node_longitude(self, times: np.ndarray) -> np.ndarray:
        """Longitude of the ascending node at ``times`` [rad]."""
        return np.deg2rad(self.ascending_node_longitude) + (
            self.nodal_precession_rate - SIDEREAL_RATE
        ) * self._elapsed(times)


def limb_look_vectors(
    observer: np.ndarray,
    along_track: np.ndarray,
    tangent_altitudes: np.ndarray,
    look_azimuth: float = 180.0,
) -> np.ndarray:
    """
    Look vectors from ``observer`` to each of ``tangent_altitudes``, in one vertical
    plane.

    Parameters
    ----------
    observer : np.ndarray
        Spacecraft position, ECEF [m]
    along_track : np.ndarray
        Spacecraft direction of motion, from OrbitState.along_track
    tangent_altitudes : np.ndarray
        Geometric tangent altitudes of the lines of sight [m]
    look_azimuth : float, optional
        Azimuth of the plane from the direction of motion, clockwise looking down
        [degrees]: 0 looks ahead along the orbit and 180 behind it. By default 180

    Returns
    -------
    np.ndarray
        Unit look vectors, (len(tangent_altitudes), 3), ECEF
    """
    geodetic = sk.WGS84()
    horizontal = _look_direction(geodetic, observer, along_track, look_azimuth)
    return np.array(
        [
            geodetic.from_tangent_altitude(altitude, observer, horizontal)
            for altitude in np.atleast_1d(tangent_altitudes)
        ]
    )


def tangent_point_geometry(
    orbit: CircularOrbit,
    time,
    look_azimuth: float = 180.0,
    tangent_altitude: float = 25000.0,
    across_track_angles: np.ndarray | None = None,
    boresight_tangent_altitude: float | None = None,
    roll: float = 0.0,
) -> xr.Dataset:
    """
    The tangent point at ``tangent_altitude`` at each of ``time``, and the solar
    geometry there.

    Without ``across_track_angles`` these are the line of sight and solar angles an
    orbit observation (hawcsimulator.steps.limb_observation.observation__orbit) has
    at that tangent altitude, for many times at once and without any radiance. With
    them, they are an imager's columns at those across-track field angles, as
    aliprocessing.geometry.InstrumentFrame defines them: the boresight looks at
    ``boresight_tangent_altitude`` in the vertical plane at ``look_azimuth``, the
    instrument's up axis is the local vertical at the spacecraft perpendicular to
    the boresight, rolled clockwise by ``roll`` looking along the boresight, and each
    column's line of sight reaches ``tangent_altitude``.

    Parameters
    ----------
    orbit : CircularOrbit
    time
        Times; without a timezone they are UTC
    look_azimuth : float, optional
        As in limb_look_vectors, by default 180
    tangent_altitude : float, optional
        Tangent altitude of the lines of sight [m], by default 25000
    across_track_angles : np.ndarray | None, optional
        Across-track field angles of the columns [degrees], positive to the right
        looking along the boresight, by default None for the boresight's plane only
    boresight_tangent_altitude : float | None, optional
        Tangent altitude of the boresight [m], by default ``tangent_altitude``
    roll : float, optional
        Roll of the instrument about the boresight [degrees], by default 0

    Returns
    -------
    xr.Dataset
        Along ``time``: spacecraft_latitude, spacecraft_longitude,
        spacecraft_altitude [m] and ascending (the boresight's tangent point is
        moving north). Along ``time``, and ``across_track`` (coordinate
        ``across_track_angle``) when given: tangent_latitude [degrees_north],
        tangent_longitude [degrees_east, -180 to 180), solar_zenith_angle [degrees],
        relative_solar_azimuth_angle [degrees, -180 to 180, 0 is forward scatter],
        scattering_angle [degrees, 0 is forward scatter, as aliprocessing's L2
        reports it] and los_azimuth_angle [degrees clockwise from north at the
        tangent point, 0 to 360)
    """
    state = orbit.state(time)
    sun = sun_directions(state.time)
    num_times = len(state.time)
    if boresight_tangent_altitude is None:
        boresight_tangent_altitude = tangent_altitude
    angles = (
        None
        if across_track_angles is None
        else np.atleast_1d(np.asarray(across_track_angles, dtype=float))
    )
    shape = (num_times,) if angles is None else (num_times, len(angles))
    point = {
        name: np.empty(shape)
        for name in (
            "tangent_latitude",
            "tangent_longitude",
            "solar_zenith_angle",
            "solar_azimuth_angle",
            "los_azimuth_angle",
        )
    }
    spacecraft = {
        name: np.empty(num_times)
        for name in (
            "spacecraft_latitude",
            "spacecraft_longitude",
            "spacecraft_altitude",
        )
    }
    ascending = np.empty(num_times, dtype=bool)
    geodetic = sk.WGS84()
    for i, (observer, along_track) in enumerate(
        zip(state.position, state.along_track, strict=True)
    ):
        # Leaves the geodetic at the spacecraft
        horizontal = _look_direction(geodetic, observer, along_track, look_azimuth)
        spacecraft["spacecraft_latitude"][i] = geodetic.latitude
        spacecraft["spacecraft_longitude"][i] = geodetic.longitude
        spacecraft["spacecraft_altitude"][i] = geodetic.altitude
        vertical = np.asarray(geodetic.local_up)

        # Leaves the geodetic at the boresight's tangent point
        boresight = np.asarray(
            geodetic.from_tangent_altitude(
                boresight_tangent_altitude, observer, horizontal
            )
        )
        # The tangent point moves with the orbit: along the spacecraft's direction
        # of motion, less its part along the tangent point's own radius
        radial = np.asarray(geodetic.location)
        radial = radial / np.linalg.norm(radial)
        ascending[i] = (along_track - (along_track @ radial) * radial)[2] > 0

        if angles is None:
            look = (
                boresight
                if boresight_tangent_altitude == tangent_altitude
                else np.asarray(
                    geodetic.from_tangent_altitude(
                        tangent_altitude, observer, horizontal
                    )
                )
            )
            looks = [(i, look)]
        else:
            looks = [
                ((i, j), look)
                for j, look in enumerate(
                    _column_looks(
                        observer,
                        boresight,
                        vertical,
                        angles,
                        roll,
                        tangent_altitude,
                    )
                )
            ]
        for index, look in looks:
            geodetic.from_tangent_point(observer, look)
            up = np.asarray(geodetic.local_up)
            north = -np.asarray(geodetic.local_south)
            east = -np.asarray(geodetic.local_west)
            point["tangent_latitude"][index] = geodetic.latitude
            point["tangent_longitude"][index] = geodetic.longitude
            point["solar_zenith_angle"][index] = np.degrees(
                np.arccos(np.clip(up @ sun[i], -1.0, 1.0))
            )
            point["solar_azimuth_angle"][index] = np.degrees(
                np.arctan2(east @ sun[i], north @ sun[i])
            )
            point["los_azimuth_angle"][index] = np.degrees(
                np.arctan2(look @ east, look @ north)
            )

    relative_azimuth = (
        point.pop("solar_azimuth_angle") - point["los_azimuth_angle"] + 180.0
    ) % 360.0 - 180.0
    zenith = np.deg2rad(point["solar_zenith_angle"])
    point["relative_solar_azimuth_angle"] = relative_azimuth
    point["scattering_angle"] = np.degrees(
        np.arccos(np.sin(zenith) * np.cos(np.deg2rad(relative_azimuth)))
    )
    point["tangent_longitude"] = (point["tangent_longitude"] + 180.0) % 360.0 - 180.0
    point["los_azimuth_angle"] = point["los_azimuth_angle"] % 360.0
    spacecraft["spacecraft_longitude"] = (
        spacecraft["spacecraft_longitude"] + 180.0
    ) % 360.0 - 180.0

    dims = ["time"] if angles is None else ["time", "across_track"]
    coords = {"time": state.time}
    if angles is not None:
        coords["across_track_angle"] = (["across_track"], angles)
    return xr.Dataset(
        {
            **{name: (dims, values) for name, values in point.items()},
            **{name: (["time"], values) for name, values in spacecraft.items()},
            "ascending": (["time"], ascending),
        },
        coords=coords,
    )


def time_at_tangent_latitude(
    orbit: CircularOrbit,
    latitude: float,
    after,
    ascending: bool = True,
    look_azimuth: float = 180.0,
    tangent_altitude: float = 25000.0,
) -> pd.Timestamp:
    """
    The first time from ``after`` that the tangent point crosses ``latitude``.

    Parameters
    ----------
    orbit : CircularOrbit
    latitude : float
        Tangent latitude [degrees north]
    after
        Time to start looking from
    ascending : bool, optional
        Find the crossing with the tangent point moving north (True) or south
        (False). On the nominal orbit the northbound crossing is in daylight, and
        the southbound one is dark except near the summer pole, where both are in
        daylight. By default True
    look_azimuth : float, optional
        As in limb_look_vectors, by default 180
    tangent_altitude : float, optional
        Tangent altitude of the line of sight to follow [m], by default 25000

    Returns
    -------
    pd.Timestamp
        UTC
    """
    start = utc_times(after)[0]

    def offset_latitude(seconds: float) -> float:
        time = start + np.timedelta64(round(seconds * 1e9), "ns")
        state = orbit.state(time)
        geodetic = sk.WGS84()
        horizontal = _look_direction(
            geodetic, state.position[0], state.along_track[0], look_azimuth
        )
        # Leaves the geodetic at the tangent point
        geodetic.from_tangent_altitude(tangent_altitude, state.position[0], horizontal)
        return geodetic.latitude - latitude

    # Sample just over one orbit, then refine the first crossing
    step = 10.0
    seconds = np.arange(0.0, 1.05 * orbit.nodal_period + step, step)
    values = np.array([offset_latitude(s) for s in seconds])
    if ascending:
        crossings = np.flatnonzero((values[:-1] < 0) & (values[1:] >= 0))
    else:
        crossings = np.flatnonzero((values[:-1] > 0) & (values[1:] <= 0))
    if len(crossings) == 0:
        direction = "northbound" if ascending else "southbound"
        msg = f"The tangent point does not cross {latitude} degrees {direction}"
        raise ValueError(msg)

    i = crossings[0]
    root = brentq(offset_latitude, seconds[i], seconds[i + 1], xtol=1e-3)
    return pd.Timestamp(start + np.timedelta64(round(root * 1e9), "ns"), tz="UTC")


def _look_direction(
    geodetic: sk.Geodetic,
    observer: np.ndarray,
    along_track: np.ndarray,
    look_azimuth: float,
) -> np.ndarray:
    """Horizontal unit vector at ``look_azimuth`` from the direction of motion."""
    geodetic.from_xyz(observer)
    up = np.asarray(geodetic.local_up)
    forward = along_track - (along_track @ up) * up
    forward = forward / np.linalg.norm(forward)
    # Clockwise looking down: from forward towards the right-hand side
    right = np.cross(forward, up)
    azimuth = np.deg2rad(look_azimuth)
    return np.cos(azimuth) * forward + np.sin(azimuth) * right


def _column_looks(
    observer: np.ndarray,
    boresight: np.ndarray,
    vertical: np.ndarray,
    angles: np.ndarray,
    roll: float,
    tangent_altitude: float,
) -> list[np.ndarray]:
    """
    Unit look vectors of an imager's columns at across-track field ``angles``
    [degrees], each reaching ``tangent_altitude``.

    As aliprocessing.geometry.InstrumentFrame: a pixel looks along
    normalize(boresight + tan(alpha) right + tan(beta) up), with up the local
    ``vertical`` at the spacecraft perpendicular to the boresight, rolled by
    ``roll`` [degrees] clockwise looking along the boresight.
    """
    up = vertical - (vertical @ boresight) * boresight
    up = up / np.linalg.norm(up)
    right = np.cross(boresight, up)
    roll_rad = np.deg2rad(roll)
    up, right = (
        np.cos(roll_rad) * up + np.sin(roll_rad) * right,
        np.cos(roll_rad) * right - np.sin(roll_rad) * up,
    )

    geodetic = sk.WGS84()
    looks = []
    for angle in angles:
        column = boresight + np.tan(np.deg2rad(angle)) * right

        def offset(tan_elevation, column=column):
            look = column + tan_elevation * up
            geodetic.from_tangent_point(observer, look / np.linalg.norm(look))
            return geodetic.altitude - tangent_altitude

        # Tangent altitude rises with elevation; 0.1 is about 240 km either way
        tan_elevation = brentq(offset, -0.1, 0.1, xtol=1e-10)
        look = column + tan_elevation * up
        looks.append(look / np.linalg.norm(look))
    return looks


def _utc_hours(times: np.ndarray) -> np.ndarray:
    """Hours since the start of the UTC day."""
    return (times - times.astype("datetime64[D]")) / np.timedelta64(1, "h")


# The nominal HAWC orbit: sun-synchronous at 450 km, crossing the equator northbound
# at 13:30 local time. The epoch only sets where the spacecraft is at a given time;
# time_at_tangent_latitude finds the time of an observation.
NOMINAL_ORBIT = CircularOrbit.sun_synchronous(
    altitude=450e3,
    local_time_of_ascending_node=13.5,
    ascending_node_time=np.datetime64("2026-01-01T00:00:00"),
)
