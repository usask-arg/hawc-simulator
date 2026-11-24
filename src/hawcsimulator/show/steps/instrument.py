from __future__ import annotations

import numpy as np
from showlib.cal_db import CalibrationDatabase
from showlib.l1a.data import L1ADataSet
from showlib.l1a.simulate import simulate_l1a
from showlib.l1b.data import L1bDataSet
from showlib.processing.l1a_to_l1b import process_l1a_to_l1b
from skretrieval.core.sasktranformat import SASKTRANRadiance

from hawcsimulator import definitions
from hawcsimulator.datastructures.viewinggeo import ObservationContainer
from hawcsimulator.definitions import (
    load_instrument,
    load_processing,
    merge,
    split_exposure,
)
from hawcsimulator.geometry.observation import observation_geometry, utc_datetime64
from hawcsimulator.show.definition import (
    auto_exposure,
    build_calibration_database,
    row_tangent_altitudes,
)


def instrument_definition(instrument: str) -> dict:
    """
    The SHOW instrument definition: a name shipped with hawcsimulator, e.g.
    "SHOW-0-SHS-REP1", or a path to a YAML file of the same form.
    """
    return load_instrument("hawcsimulator.show", instrument)


def processing_definition(processing: str) -> dict:
    """
    The SHOW processing version: a name shipped with hawcsimulator, e.g. "v0.1", or a
    path to a YAML file of the same form.
    """
    return load_processing("hawcsimulator.show", processing)


def acquisition_definition(
    acquisition: str, acquisition_cfg: dict | None = None
) -> dict:
    """
    The SHOW acquisition pattern: a name shipped with hawcsimulator, e.g.
    "nominal", or a path to a YAML file of the same form, with ``acquisition_cfg``
    merged in, e.g. ``{"timing": {...}}``.
    """
    return merge(
        definitions.load_acquisition("hawcsimulator.show", acquisition), acquisition_cfg
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
        ("showlib",),
        acquisition_definition,
    )


def observer_altitude(instrument_definition: dict) -> float:
    """Orbit altitude of the instrument definition [m]."""
    return float(instrument_definition["orbit_altitude_m"])


def viewing_tangent_altitudes(instrument_definition: dict) -> np.ndarray:
    """Tangent altitude of each line of sight [m]; one detector row each."""
    return row_tangent_altitudes(instrument_definition)


def calibration_database(
    instrument_definition: dict,
    viewing_tangent_altitudes: np.ndarray,
    observer_altitude: float,
    cal_db_cfg: dict | None = None,
) -> CalibrationDatabase:
    """
    The SHOW calibration database of the instrument definition, with one detector row
    per viewing tangent altitude.

    Parameters
    ----------
    cal_db_cfg : dict | None, optional
        Overrides of the settings derived from the instrument definition: arguments of
        showlib.cal_db.ideal.ideal_calibration_database, plus ``detector_gain`` and
        ``filter_peak_transmission``. By default None
    """
    return build_calibration_database(
        instrument_definition, viewing_tangent_altitudes, observer_altitude, cal_db_cfg
    )


def sample_wavelengths(calibration_database: CalibrationDatabase) -> np.ndarray:
    """L1B sample wavelengths [nm] of the first detector row."""
    return 1e7 / calibration_database.sample_wavenumber(0)


def l1a(
    front_end_radiance: SASKTRANRadiance,
    observation: ObservationContainer,
    calibration_database: CalibrationDatabase,
    instrument_definition: dict,
    acquisition_definition: dict,
    provenance: dict,
    l1a_cfg: dict | None = None,
) -> L1ADataSet:
    """
    L1A interferograms from the front end radiance, with showlib's instrument model.

    The image integrates the acquisition pattern's ``timing.integration_time_s``, or
    when that is "auto", the time that fills the brightest pixel to
    ``timing.full_well_fraction`` of full well. When the instrument definition limits
    the single exposure (``detector.coadd_exposures``), the image averages equal
    exposures that add up to it.

    Parameters
    ----------
    l1a_cfg : dict | None, optional
        Keyword arguments for showlib.l1a.simulate.simulate_l1a: noise and seed, and
        exposure_time and num_averaged_exposures to replace the acquisition pattern's
        exposure, by default None
    """
    cfg = dict(l1a_cfg or {})
    wavenumber, radiance = _radiance_by_row(front_end_radiance, calibration_database)
    if cfg.get("exposure_time") is None:
        timing = acquisition_definition["timing"]
        total = timing["integration_time_s"]
        if total == "auto":
            total = auto_exposure(
                calibration_database,
                wavenumber,
                radiance,
                timing.get("full_well_fraction", 0.8),
            )
        exposure, num_exposures = split_exposure(float(total), instrument_definition)
        cfg["exposure_time"] = exposure
        cfg.setdefault("num_averaged_exposures", num_exposures)
    geometry = observation_geometry(observation.observation)
    image_geometry = {
        "time": utc_datetime64(observation.time),
        "tangent_altitude": geometry["tangent_altitude"],
        "tangent_latitude": geometry["tangent_latitude"],
        "tangent_longitude": geometry["tangent_longitude"],
        "spacecraft_latitude": float(geometry["observer_latitude"][0]),
        "spacecraft_longitude": float(geometry["observer_longitude"][0]),
        "spacecraft_altitude": float(geometry["observer_altitude"][0]),
        "solar_zenith_angle": geometry["solar_zenith_angle"],
        "relative_solar_azimuth_angle": geometry["relative_solar_azimuth_angle"],
        "los_azimuth_angle": geometry["los_azimuth_angle"],
    }
    result = simulate_l1a(
        calibration_database,
        wavenumber,
        radiance[np.newaxis],
        [image_geometry],
        **cfg,
    )
    result.ds.attrs.update(provenance)
    return result


def l1b(
    l1a: L1ADataSet,
    calibration_database: CalibrationDatabase,
    processing_definition: dict,
    provenance: dict,
    l1b_cfg: dict | None = None,
) -> L1bDataSet:
    """
    L1B radiance spectra from the L1A interferograms, with the processing version's
    settings.

    Parameters
    ----------
    l1b_cfg : dict | None, optional
        Overrides of the processing version's keyword arguments for
        showlib.processing.l1a_to_l1b.process_l1a_to_l1b: apodization, noise_fraction
        and dark. By default None
    """
    settings = merge(processing_definition["l1b"], l1b_cfg)
    result = process_l1a_to_l1b(l1a, calibration_database, **settings)
    result.ds.attrs.update(provenance)
    return result


def _radiance_by_row(
    front_end_radiance: SASKTRANRadiance, calibration_database: CalibrationDatabase
) -> tuple[np.ndarray, np.ndarray]:
    """Increasing wavenumber, and Stokes I radiance with shape (row, wavenumber)."""
    data = front_end_radiance.data
    radiance = data["radiance"]
    if "stokes" in radiance.dims:
        radiance = radiance.isel(stokes=0)
    spectral_dim = next(dim for dim in radiance.dims if dim != "los")

    wavenumber = data["wavenumber_cminv"].to_numpy()
    order = np.argsort(wavenumber)
    wavenumber = wavenumber[order]
    radiance = radiance.transpose("los", spectral_dim).to_numpy()[:, order]

    model_wavenumber = calibration_database.model_wavenumber()
    if wavenumber.shape != model_wavenumber.shape or not np.allclose(
        wavenumber, model_wavenumber, rtol=0, atol=1e-6
    ):
        msg = (
            "The front end radiance must be on the calibration database's model "
            "wavenumber grid, calibration_database.model_wavenumber()"
        )
        raise ValueError(msg)
    if radiance.shape[0] != calibration_database.num_rows:
        msg = (
            f"The front end radiance has {radiance.shape[0]} lines of sight, but the "
            f"calibration database has {calibration_database.num_rows} rows"
        )
        raise ValueError(msg)
    return wavenumber, radiance
