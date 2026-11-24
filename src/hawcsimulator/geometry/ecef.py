from __future__ import annotations

import numpy as np
import sasktran2 as sk
import xarray as xr
from sasktran2.viewinggeo import ecef_to_sasktran2_ray
from sasktran2.viewinggeo.base import ViewingGeometryContainer

from hawcsimulator.geometry.solar import SunDirectionSolarHandler
from hawcsimulator.geometry.utc import utc_times


class ECEFLimbGeometry(sk.ViewingGeometry, ViewingGeometryContainer):
    """
    Limb lines of sight given by their time, observer position and look direction in
    ECEF coordinates.

    Each line of sight becomes a sasktran2 ray through
    sasktran2.viewinggeo.ecef_to_sasktran2_ray, with the solar angles at its own
    tangent point. Lines of sight that end at the ground stay limb rays with a
    negative tangent altitude, which sasktran2 ends at the surface.

    ``geometry_ds`` holds, along ``los``: ``time``, ``observer_position_ecef`` and
    ``look_direction_ecef`` (with ``xyz``), which define the geometry, and
    tangent_altitude [m], tangent_latitude [degrees_north], tangent_longitude
    [degrees_east, -180 to 180), observer_latitude, observer_longitude,
    observer_altitude [m], solar_zenith_angle and solar_azimuth_angle [degrees, at
    the tangent point, the azimuth clockwise from north towards the sun],
    los_azimuth_angle [degrees clockwise from north at the tangent point, 0 to 360)
    and relative_solar_azimuth_angle [solar minus line of sight azimuth, degrees,
    -180 to 180, 0 is forward scatter].

    Parameters
    ----------
    time
        Time of each line of sight, or one time for all of them. Times without a
        timezone are UTC.
    observer_position : np.ndarray
        Observer positions, (los, 3), or one position (3,) for all of them, ECEF [m]
    look_direction : np.ndarray
        Look vectors, (los, 3), ECEF. They are normalized.
    solar_handler : sk.solar.SolarGeometryHandlerBase | None, optional
        Solar angles for each tangent point, by default SunDirectionSolarHandler
    forced_solar_angles : tuple[float, float] | None, optional
        Solar zenith angle and relative solar azimuth angle [degrees, 0 is forward
        scatter] to give every line of sight at its tangent point instead of the
        sun's, by default None
    reference_altitude : float, optional
        Tangent altitude the model geometry's solar zenith angle and Earth radius
        are taken at [m], by default 25000
    """

    def __init__(
        self,
        time,
        observer_position: np.ndarray,
        look_direction: np.ndarray,
        solar_handler: sk.solar.SolarGeometryHandlerBase | None = None,
        forced_solar_angles: tuple[float, float] | None = None,
        reference_altitude: float = 25000.0,
    ) -> None:
        look = np.atleast_2d(np.asarray(look_direction, dtype=float))
        num_los = len(look)
        if look.shape != (num_los, 3):
            msg = f"look_direction must have shape (los, 3), got {look.shape}"
            raise ValueError(msg)
        look = look / np.linalg.norm(look, axis=1, keepdims=True)
        observer = np.broadcast_to(
            np.asarray(observer_position, dtype=float), (num_los, 3)
        ).copy()
        times = np.broadcast_to(utc_times(time), (num_los,)).copy()
        if solar_handler is None:
            solar_handler = SunDirectionSolarHandler()
        self._reference_altitude = reference_altitude

        sk.ViewingGeometry.__init__(self)

        names = (
            "tangent_altitude",
            "tangent_latitude",
            "tangent_longitude",
            "observer_latitude",
            "observer_longitude",
            "observer_altitude",
            "solar_zenith_angle",
            "solar_azimuth_angle",
            "los_azimuth_angle",
            "earth_radius",
        )
        fields = {name: np.empty(num_los) for name in names}
        geodetic = sk.WGS84()
        for i in range(num_los):
            geodetic.from_xyz(observer[i])
            fields["observer_latitude"][i] = geodetic.latitude
            fields["observer_longitude"][i] = geodetic.longitude
            fields["observer_altitude"][i] = geodetic.altitude

            geodetic.from_tangent_point(observer[i], look[i])
            up = np.asarray(geodetic.local_up)
            north = -np.asarray(geodetic.local_south)
            east = -np.asarray(geodetic.local_west)
            los_azimuth = np.degrees(np.arctan2(look[i] @ east, look[i] @ north))
            ray_solar_handler = solar_handler
            if forced_solar_angles is not None:
                # The forced relative azimuth is measured from this line of sight
                forced_zenith, forced_relative_azimuth = forced_solar_angles
                ray_solar_handler = sk.solar.SolarGeometryHandlerForced(
                    forced_zenith, (los_azimuth + forced_relative_azimuth) % 360.0
                )
            solar_zenith, solar_azimuth = ray_solar_handler.target_solar_angles(
                geodetic.latitude, geodetic.longitude, geodetic.altitude, times[i]
            )
            fields["tangent_altitude"][i] = geodetic.altitude
            fields["tangent_latitude"][i] = geodetic.latitude
            fields["tangent_longitude"][i] = geodetic.longitude
            fields["solar_zenith_angle"][i] = solar_zenith
            fields["solar_azimuth_angle"][i] = solar_azimuth
            fields["los_azimuth_angle"][i] = los_azimuth

            self.add_ray(
                ecef_to_sasktran2_ray(
                    observer[i],
                    look[i],
                    times[i],
                    geoid=sk.WGS84(),
                    solar_handler=ray_solar_handler,
                    ground_elevation=-np.inf,
                )
            )
            # Radius of the surface below the tangent point, as LimbVertical uses
            fields["earth_radius"][i] = np.linalg.norm(
                np.asarray(geodetic.location) - geodetic.altitude * up
            )

        fields["tangent_longitude"] = _wrap_180(fields["tangent_longitude"])
        fields["observer_longitude"] = _wrap_180(fields["observer_longitude"])
        fields["los_azimuth_angle"] = fields["los_azimuth_angle"] % 360.0
        fields["relative_solar_azimuth_angle"] = _wrap_180(
            fields["solar_azimuth_angle"] - fields["los_azimuth_angle"]
        )

        geometry_ds = xr.Dataset(
            {
                "observer_position_ecef": (["los", "xyz"], observer),
                "look_direction_ecef": (["los", "xyz"], look),
                **{name: (["los"], values) for name, values in fields.items()},
            },
            coords={"time": (["los"], times), "xyz": ["x", "y", "z"]},
        )
        ViewingGeometryContainer.__init__(self, geometry_ds)

    def recommended_cos_sza(self) -> float:
        """Cosine of the solar zenith angle at the reference tangent altitude."""
        return float(
            np.cos(np.deg2rad(self._at_reference_altitude("solar_zenith_angle")))
        )

    def recommended_earth_radius(self) -> float:
        """Earth radius below the tangent point at the reference tangent altitude [m]."""
        return self._at_reference_altitude("earth_radius")

    def model_geometry(self, altitude_grid_m: np.ndarray) -> sk.Geometry1D:
        return sk.Geometry1D(
            self.recommended_cos_sza(),
            0.0,
            self.recommended_earth_radius(),
            altitude_grid_m,
            sk.InterpolationMethod.LinearInterpolation,
            sk.GeometryType.Spherical,
        )

    def _at_reference_altitude(self, name: str) -> float:
        altitude = self.geometry_ds["tangent_altitude"].to_numpy()
        order = np.argsort(altitude)
        values = self.geometry_ds[name].to_numpy()
        return float(
            np.interp(self._reference_altitude, altitude[order], values[order])
        )


def _wrap_180(angle):
    return (np.asarray(angle) + 180.0) % 360.0 - 180.0
