from __future__ import annotations

import numpy as np
import sasktran2 as sk
import xarray as xr
from aliprocessing.cal_db import SimpleInstrument
from aliprocessing.forward_model import (
    exposure_for_fill,
    observation_sequence,
    scene_from_radiance,
    simulate_l1a,
)
from aliprocessing.geometry import boresight_elevation_for_tangent_altitude
from aliprocessing.processing.l1a_to_l1b import process_l1a_to_l1b
from skretrieval.core.sasktranformat import SASKTRANRadiance

from hawcsimulator import definitions
from hawcsimulator.ali.definition import build_calibration_database, simple_instrument
from hawcsimulator.datastructures.viewinggeo import ObservationContainer
from hawcsimulator.definitions import (
    dead_time,
    load_instrument,
    load_processing,
    merge,
    split_exposure,
)
from hawcsimulator.geometry.observation import observation_geometry, utc_datetime64


def instrument_definition(instrument: str) -> dict:
    """
    The ALI instrument definition: a name shipped with hawcsimulator, e.g.
    "ALI-0-FWHEEL-REP1", or a path to a YAML file of the same form.
    """
    return load_instrument("hawcsimulator.ali", instrument)


def processing_definition(processing: str) -> dict:
    """
    The ALI processing version: a name shipped with hawcsimulator, e.g. "v0.1", or a
    path to a YAML file of the same form.
    """
    return load_processing("hawcsimulator.ali", processing)


def acquisition_definition(
    acquisition: str, acquisition_cfg: dict | None = None
) -> dict:
    """
    The ALI acquisition pattern: a name shipped with hawcsimulator, e.g.
    "nominal", or a path to a YAML file of the same form, with ``acquisition_cfg``
    merged in, e.g. ``{"timing": {...}}``.
    """
    return merge(
        definitions.load_acquisition("hawcsimulator.ali", acquisition), acquisition_cfg
    )


def provenance(
    instrument_definition: dict,
    processing_definition: dict,
    acquisition_definition: dict,
) -> dict:
    """
    Attributes recording the instrument, processing version, acquisition pattern and
    package versions.
    """
    return definitions.provenance(
        instrument_definition,
        processing_definition,
        ("aliprocessing",),
        acquisition_definition,
    )


def observer_altitude(instrument_definition: dict) -> float:
    """Orbit altitude of the instrument definition [m]."""
    return float(instrument_definition["orbit_altitude_m"])


def ali_instrument(
    instrument_definition: dict, cal_db_cfg: dict | None = None
) -> SimpleInstrument:
    """
    aliprocessing's SimpleInstrument for the instrument definition.

    Parameters
    ----------
    cal_db_cfg : dict | None, optional
        Overrides of SimpleInstrument fields derived from the instrument definition,
        e.g. ``{"center_wavelengths_nm": (745.0, 1020.0)}``. By default None
    """
    return simple_instrument(instrument_definition, cal_db_cfg)


def calibration_database(
    ali_instrument: SimpleInstrument, instrument_definition: dict
) -> xr.Dataset:
    """
    The ALI calibration database: aliprocessing's simple instrument model with the
    detector's quantum efficiency curve. Pass a different database through
    ``overrides``.
    """
    return build_calibration_database(instrument_definition, ali_instrument)


def sample_wavelengths(calibration_database: xr.Dataset) -> np.ndarray:
    """Centre wavelengths of the spectral channels [nm]."""
    return calibration_database["center_wavelength"].to_numpy()


def pointing(observation: ObservationContainer, acquisition_definition: dict) -> dict:
    """
    Spacecraft position and boresight pointing.

    The spacecraft is the observation's observer. The boresight looks along the
    observation's lines of sight, raised or lowered to the acquisition pattern's
    ``pointing.boresight_tangent_altitude_m`` and rolled by
    ``pointing.boresight_roll_deg``, the same for every observation set.

    Returns
    -------
    dict
        spacecraft_latitude, spacecraft_longitude, spacecraft_altitude,
        boresight_azimuth, boresight_elevation and boresight_roll, in degrees and
        metres
    """
    settings = acquisition_definition["pointing"]
    tangent_altitude = float(settings["boresight_tangent_altitude_m"])
    roll = float(settings.get("boresight_roll_deg", 0.0))

    geometry = observation_geometry(observation.observation)
    spacecraft = {
        "spacecraft_latitude": float(geometry["observer_latitude"][0]),
        "spacecraft_longitude": float(geometry["observer_longitude"][0]),
        "spacecraft_altitude": float(geometry["observer_altitude"][0]),
    }

    observer = sk.WGS84()
    observer.from_lat_lon_alt(
        spacecraft["spacecraft_latitude"],
        spacecraft["spacecraft_longitude"],
        spacecraft["spacecraft_altitude"],
    )
    i = int(np.argmin(np.abs(geometry["tangent_altitude"] - tangent_altitude)))
    tangent = sk.WGS84()
    tangent.from_lat_lon_alt(
        float(geometry["tangent_latitude"][i]),
        float(geometry["tangent_longitude"][i]),
        float(geometry["tangent_altitude"][i]),
    )
    look = np.asarray(tangent.location) - np.asarray(observer.location)
    north = -np.asarray(observer.local_south)
    east = -np.asarray(observer.local_west)
    azimuth = float(np.degrees(np.arctan2(look @ east, look @ north)) % 360.0)

    elevation = float(
        boresight_elevation_for_tangent_altitude(
            spacecraft["spacecraft_latitude"],
            spacecraft["spacecraft_longitude"],
            spacecraft["spacecraft_altitude"],
            azimuth,
            tangent_altitude,
        )
    )
    return {
        **spacecraft,
        "boresight_azimuth": azimuth,
        "boresight_elevation": elevation,
        "boresight_roll": roll,
    }


def scene(
    front_end_radiance: SASKTRANRadiance,
    observation: ObservationContainer,
    calibration_database: xr.Dataset,
) -> xr.Dataset:
    """
    The front end radiance as aliprocessing's scene: I, Q and U against tangent
    altitude for each spectral channel, from
    aliprocessing.forward_model.scene_from_radiance.

    Every pixel sees the scene at its own tangent altitude, so the observation's
    tangent altitudes must cover the L1B altitude grid widened by the resampling
    kernel; aliprocessing checks this.
    """
    data = front_end_radiance.data
    radiance = (
        data["radiance"]
        .rename(spectral_grid="wavelength")
        .assign_coords(wavelength=("wavelength", data["wavelength_nm"].to_numpy()))
    )
    tangent_altitude = observation_geometry(observation.observation)["tangent_altitude"]
    return scene_from_radiance(calibration_database, radiance, tangent_altitude)


def l1a(
    scene: xr.Dataset,
    pointing: dict,
    observation: ObservationContainer,
    calibration_database: xr.Dataset,
    instrument_definition: dict,
    acquisition_definition: dict,
    provenance: dict,
    l1a_cfg: dict | None = None,
) -> xr.DataTree:
    """
    L1A detector frames from the scene, with aliprocessing's instrument model.

    The acquisition pattern's ``timing`` sets the frames: ``num_sets`` observation
    sets that each step through every spectral channel, with a blank frame after each
    set when ``blank_between_sets``. Each frame integrates ``exposure`` seconds (one
    value or one per channel), or with "auto" the time that fills the brightest pixel
    to ``fill`` of saturation. When the instrument definition limits the single
    exposure (``detector.coadd_exposures``), a frame averages equal exposures that add
    up to it. Each frame takes its exposures plus the detector's dead time after each
    (``detector.dead_time_s``), and the next starts ``filter_change_s`` later. The
    pointing of each set comes from ``pointing``.

    Parameters
    ----------
    l1a_cfg : dict | None, optional
        ``noise`` (default True) and ``seed`` for the noise, and ``exposure_time``
        and ``num_averaged_exposures`` to replace the acquisition pattern's
        exposure: the single-exposure time, one value or one per channel, and the
        number of exposures each frame averages. By default None
    """
    cfg = {"noise": True, "seed": None}
    cfg.update(l1a_cfg or {})
    noise = cfg.pop("noise")
    seed = cfg.pop("seed")
    exposure = cfg.pop("exposure_time", None)
    num_averaged = cfg.pop("num_averaged_exposures", None)
    if cfg:
        msg = (
            f"Unknown l1a_cfg keys {sorted(cfg)}; the acquisition pattern sets the "
            "timing and pointing, adjust it with acquisition_cfg"
        )
        raise ValueError(msg)

    timing = acquisition_definition["timing"]
    num_channels = calibration_database.sizes["spectral_channel"]
    if exposure is None:
        total = timing["exposure"]
        if total == "auto":
            total = exposure_for_fill(
                calibration_database, scene, timing.get("fill", 0.6)
            )
        total = np.broadcast_to(np.asarray(total, dtype=float), (num_channels,))
        exposure, coadds = np.array(
            [split_exposure(t, instrument_definition) for t in total]
        ).T
    else:
        exposure = np.broadcast_to(np.asarray(exposure, dtype=float), (num_channels,))
        coadds = np.full(num_channels, 1 if num_averaged is None else num_averaged)
    exposure = np.asarray(exposure, dtype=float)
    coadds = np.asarray(coadds).astype(int)

    frames = observation_sequence(
        calibration_database,
        start=utc_datetime64(observation.time),
        num_sets=int(timing["num_sets"]),
        exposure_time=exposure,
        spacecraft_latitude=pointing["spacecraft_latitude"],
        spacecraft_longitude=pointing["spacecraft_longitude"],
        spacecraft_altitude=pointing["spacecraft_altitude"],
        boresight_azimuth=pointing["boresight_azimuth"],
        boresight_elevation=pointing["boresight_elevation"],
        boresight_roll=pointing["boresight_roll"],
        blank_between_sets=bool(timing.get("blank_between_sets", True)),
    )
    frames = _schedule(
        frames,
        coadds,
        dead_time(instrument_definition),
        float(timing.get("filter_change_s", 0.0)),
    )
    result = simulate_l1a(calibration_database, scene, frames, noise=noise, seed=seed)
    result.attrs.update(provenance)
    return result


def _schedule(
    frames: xr.Dataset,
    coadds: np.ndarray,
    dead_time: float,
    filter_change: float,
) -> xr.Dataset:
    """
    Each frame's number of exposures and its start time: after the previous frame's
    exposures, each followed by the dead time, and a filter change.
    """
    channel = frames["spectral_channel"].to_numpy()
    exposure = frames["exposure_time"].to_numpy()
    # Blank frames use the longest single exposure, so that channel's number
    blank = coadds[int(np.argmax(np.where(channel >= 0, exposure, -np.inf)))]
    num_exposures = np.where(channel >= 0, coadds[np.maximum(channel, 0)], blank)

    duration = num_exposures * (exposure + dead_time) + filter_change
    offset = np.concatenate([[0.0], np.cumsum(duration[:-1])])
    start = frames["time"].to_numpy()[0]
    time = start + np.round(offset * 1e9).astype("timedelta64[ns]")

    def per_frame(name, values):
        return frames[name].copy(data=np.asarray(values, dtype=frames[name].dtype))

    frames = frames.assign(
        num_averaged_exposures=per_frame("num_averaged_exposures", num_exposures),
    )
    return frames.assign_coords(time=frames["time"].copy(data=time))


def l1b(
    l1a: xr.DataTree,
    calibration_database: xr.Dataset,
    processing_definition: dict,
    provenance: dict,
    tangent_solar_zenith_angle: float | None = None,
    tangent_solar_azimuth_angle: float | None = None,
    l1b_cfg: dict | None = None,
) -> xr.Dataset:
    """
    L1B radiance and polarization from the L1A frames.

    L1A to L1B computes the sun's position from the observation time. When the
    observation's solar angles are forced instead, the L1B solar angles are
    replaced with them, so the L2 retrieval sees the same sun as the front end
    radiance.

    Parameters
    ----------
    l1b_cfg : dict | None, optional
        Overrides of the processing version's keyword arguments for
        aliprocessing.processing.l1a_to_l1b.process_l1a_to_l1b, e.g. ``dark``, by
        default None
    """
    settings = merge(processing_definition["l1b"], l1b_cfg)
    l1b = process_l1a_to_l1b(l1a, calibration_database, **settings)
    l1b.attrs.update(provenance)

    if (
        tangent_solar_zenith_angle is not None
        and tangent_solar_azimuth_angle is not None
    ):
        relative_azimuth = (tangent_solar_azimuth_angle + 180.0) % 360.0 - 180.0
        l1b = l1b.assign(
            solar_zenith_angle=xr.full_like(
                l1b["solar_zenith_angle"], tangent_solar_zenith_angle
            ),
            relative_solar_azimuth_angle=xr.full_like(
                l1b["relative_solar_azimuth_angle"], relative_azimuth
            ),
        )
    return l1b
