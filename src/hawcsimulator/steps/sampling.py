from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from hawcsimulator.geometry.orbit import (
    NOMINAL_ORBIT,
    CircularOrbit,
    tangent_point_geometry,
)
from hawcsimulator.geometry.utc import utc_times
from hawcsimulator.steps.limb_observation import REFERENCE_ALTITUDE


def sampling(
    sampling_start: pd.Timestamp,
    sampling_end: pd.Timestamp,
    acquisition_definition: dict,
    observation_duration: float | None,
    provenance: dict,
    sampling_across_track_angles: np.ndarray | None = None,
    orbit: CircularOrbit = NOMINAL_ORBIT,
    look_azimuth: float = 180.0,
    maximum_allowed_sza: float = 88,
) -> xr.Dataset:
    """
    When and where the instrument measures from ``sampling_start`` to
    ``sampling_end``, without simulating any measurement.

    Observations start every ``sampling.interval_s`` of the acquisition pattern from
    ``sampling_start``, and those whose tangent point at 25 km is in daylight (solar
    zenith angle at most ``maximum_allowed_sza``) are kept. Each has the geometry an
    orbit observation (``config={"observation_method": "orbit"}``) at that time
    would have at 25 km. An imager with across-track bins (ALI) has a tangent point
    for each bin: the columns at those across-track angles, with the boresight and
    roll of the acquisition pattern's ``pointing``.

    Parameters
    ----------
    sampling_start, sampling_end : pd.Timestamp
        The period to sample; times without a timezone are UTC
    sampling_across_track_angles : np.ndarray | None, optional
        Across-track angles of the profiles in each observation [degrees], by default
        None for one line of sight
    orbit : CircularOrbit, optional
        The spacecraft orbit, by default the nominal HAWC orbit
    look_azimuth : float, optional
        Azimuth of the lines of sight from the spacecraft's direction of motion in
        [degrees], by default 180, behind the spacecraft
    maximum_allowed_sza : float, optional
        Largest solar zenith angle at 25 km in [degrees], by default 88

    Returns
    -------
    xr.Dataset
        Along ``time``, the start of each observation, and ``across_track`` for an
        imager, the variables of hawcsimulator.geometry.orbit.tangent_point_geometry,
        with the provenance and ``sampling_interval_s`` in its attributes
    """
    interval = float(acquisition_definition["sampling"]["interval_s"])
    if observation_duration is not None and interval < observation_duration:
        msg = (
            f"The acquisition pattern's sampling interval, {interval} s, is shorter "
            f"than one observation, {observation_duration:.3f} s"
        )
        raise ValueError(msg)

    start = utc_times(sampling_start)[0]
    end = utc_times(sampling_end)[0]
    period = (end - start) / np.timedelta64(1, "s")
    offsets = np.arange(0.0, period, interval)
    times = start + np.round(offsets * 1e9).astype("timedelta64[ns]")

    samples = tangent_point_geometry(orbit, times, look_azimuth, REFERENCE_ALTITUDE)
    samples = samples.isel(
        time=(samples["solar_zenith_angle"] <= maximum_allowed_sza).to_numpy()
    )
    if sampling_across_track_angles is not None:
        pointing = acquisition_definition.get("pointing", {})
        samples = tangent_point_geometry(
            orbit,
            samples["time"].to_numpy(),
            look_azimuth,
            REFERENCE_ALTITUDE,
            across_track_angles=sampling_across_track_angles,
            boresight_tangent_altitude=float(
                pointing.get("boresight_tangent_altitude_m", REFERENCE_ALTITUDE)
            ),
            roll=float(pointing.get("boresight_roll_deg", 0.0)),
        )
    samples.attrs.update(provenance)
    samples.attrs["sampling_interval_s"] = interval
    return samples
