"""SHOW instrument definitions as showlib calibration databases."""

from __future__ import annotations

import numpy as np
from showlib.cal_db import CalibrationDatabase, bin_width
from showlib.cal_db.ideal import ideal_calibration_database

from hawcsimulator.definitions import load_detector, read_noise

EARTH_RADIUS = 6371e3

# Spectral transmission table of the calibration database: how far it extends past
# the filter edges [nm] and its spacing [cm-1]
TABLE_MARGIN_NM = 0.5
TABLE_RESOLUTION_CMINV = 0.02

# Calibration database settings that showlib's ideal_calibration_database does not
# take, applied after it builds the database
_POST_BUILD = ("detector_gain", "filter_peak_transmission")


def _optical_layout(definition: dict, detector: dict, num_rows: int) -> dict:
    """
    The spectral and radiometric layout of an instrument definition: number of
    columns, OPD spacing, Littrow wavenumber, detector rows binned into each line of
    sight, effective collecting area (aperture times every efficiency), fringe
    visibility, and an optional grating tilt.

    A design definition describes the interferometer (``interferometer``,
    ``aperture``, ``efficiency``); a representative one gives the spectral
    performance and one effective throughput (``spectral``, ``throughput`` and
    ``detector.rows_per_line_of_sight``).
    """
    pitch_cm = detector["pixel_pitch_m"] * 100
    if "interferometer" in definition:
        interferometer = definition["interferometer"]
        magnification = interferometer["grating_to_detector_magnification"]
        efficiency = definition["efficiency"]
        height = interferometer["illuminated_grating_height_cm"]
        return {
            "num_columns": round(
                interferometer["illuminated_grating_width_cm"]
                / magnification
                / pitch_cm
            ),
            "opd_spacing_cm": interferometer["opd_per_grating_cm"]
            * pitch_cm
            * magnification,
            "littrow_wavenumber_cminv": interferometer["littrow_wavenumber_cminv"],
            "binning": height / magnification / pitch_cm / num_rows,
            "effective_area_m2": definition["aperture"]["width_m"]
            * definition["aperture"]["height_m"]
            * efficiency["front_optics"]
            * efficiency["back_optics"]
            * efficiency["quantum_efficiency"]
            * interferometer["output_fraction"],
            "fringe_visibility": interferometer["fringe_visibility"],
            "grating_tilt_rad": interferometer["grating_tilt_rad"],
            "grating_height_cm": height,
        }

    spectral = definition["spectral"]
    num_columns = int(spectral["num_samples"])
    # The OPD runs over +-num_columns/2 samples about zero path difference, and the
    # resolution is 1 / (2 * maximum OPD)
    rows_per_line_of_sight = definition["detector"]["rows_per_line_of_sight"]
    return {
        "num_columns": num_columns,
        "opd_spacing_cm": 1.0 / (spectral["resolution_cminv"] * num_columns),
        "littrow_wavenumber_cminv": spectral["littrow_wavenumber_cminv"],
        "binning": rows_per_line_of_sight
        * definition["viewing"]["num_rows"]
        / num_rows,
        "effective_area_m2": definition["throughput"]["effective_area_m2"],
        "fringe_visibility": spectral.get("fringe_visibility", 1.0),
        "grating_tilt_rad": 0.0,
        "grating_height_cm": None,
    }


def calibration_database_kwargs(
    definition: dict, tangent_altitudes: np.ndarray, observer_altitude: float
) -> dict:
    """
    Settings of the calibration database for an instrument definition: arguments of
    showlib.cal_db.ideal.ideal_calibration_database, plus ``detector_gain`` and
    ``filter_peak_transmission``.

    Each line of sight is one row, a bin of detector rows; detector noise, full well
    and dark signal are those of the binned pixel. Each column is one interferogram
    sample.
    """
    detector = load_detector(definition["detector"]["model"])
    mode = detector["gain_modes"][definition["detector"]["gain_mode"]]
    num_rows = len(tangent_altitudes)
    layout = _optical_layout(definition, detector, num_rows)
    num_columns = layout["num_columns"]
    binning = layout["binning"]

    gain = mode["full_well_e"] / 2 ** detector["adc_bits"]
    # The binned pixel's counts are summed, so its ADC range is binning times wider
    adc_bits = int(np.ceil(np.log2(binning * 2 ** detector["adc_bits"])))

    # Each line of sight's vertical field of view: the tangent-altitude spacing of the
    # rows, seen from the observer at the middle of the field of view
    viewing = definition["viewing"]
    tangent_altitudes = np.asarray(tangent_altitudes, dtype=float)
    if num_rows > 1:
        row_height = (tangent_altitudes.max() - tangent_altitudes.min()) / (
            num_rows - 1
        )
    else:
        low, high = viewing["tangent_altitude_range_m"]
        row_height = high - low
    limb_distance = np.sqrt(
        (EARTH_RADIUS + observer_altitude) ** 2
        - (EARTH_RADIUS + tangent_altitudes.mean()) ** 2
    )
    row_fov = row_height / limb_distance
    across_track_fov = np.deg2rad(viewing["across_track_fov_deg"])
    # The interferogram spreads each line of sight's light across the columns
    pixel_etendue = (
        layout["effective_area_m2"] * across_track_fov * row_fov / num_columns
    )

    dark = definition["detector"]
    filter_ = definition["filter"]
    kwargs = {
        "num_columns": num_columns,
        "opd_spacing_cm": layout["opd_spacing_cm"],
        "littrow_wavelength_nm": 1e7 / layout["littrow_wavenumber_cminv"],
        "filter_centre_nm": filter_["centre_wavelength_nm"],
        "filter_half_width_nm": (filter_["fwhm_nm"] - filter_["edge_width_nm"]) / 2,
        "filter_transition_nm": filter_["edge_width_nm"],
        "filter_stopband_db": filter_["stopband_db"],
        "min_sample_transmission": definition["min_sample_transmission"],
        "table_margin_nm": TABLE_MARGIN_NM,
        "table_resolution_cminv": TABLE_RESOLUTION_CMINV,
        "pixel_etendue": pixel_etendue,
        "detector_full_well": mode["full_well_e"] * binning,
        "detector_adc_bits": adc_bits,
        "detector_read_noise": read_noise(definition) * np.sqrt(binning),
        "dark_offset": dark["dark_offset_dn"] * binning,
        "dark_offset_spread": dark["dark_offset_spread_dn"] * np.sqrt(binning),
        "dark_current": detector["dark_current_e_per_s"] * binning / gain,
        "fringe_visibility": layout["fringe_visibility"],
        "valid_from": definition["valid_from"],
        "detector_gain": gain,
        "filter_peak_transmission": filter_["peak_transmission"],
    }
    if layout["grating_tilt_rad"]:
        height = layout["grating_height_cm"]
        kwargs["grating_tilt_rad"] = layout["grating_tilt_rad"]
        kwargs["row_edge_height_cm"] = np.linspace(
            -height / 2, height / 2, num_rows + 1
        )
    return kwargs


def build_calibration_database(
    definition: dict,
    tangent_altitudes: np.ndarray,
    observer_altitude: float,
    overrides: dict | None = None,
) -> CalibrationDatabase:
    """
    The calibration database of an instrument definition.

    ``overrides`` replace settings of calibration_database_kwargs, e.g.
    ``{"dark_current": 0.0}``.
    """
    kwargs = calibration_database_kwargs(
        definition, tangent_altitudes, observer_altitude
    )
    kwargs.update(overrides or {})
    post_build = {name: kwargs.pop(name) for name in _POST_BUILD}

    ds = ideal_calibration_database(len(tangent_altitudes), **kwargs).ds.copy()
    ds["detector_gain"] = ds["detector_gain"].copy(
        data=np.float64(post_build["detector_gain"])
    )
    ds["spectral_transmission"] = ds["spectral_transmission"].copy(
        data=ds["spectral_transmission"].to_numpy()
        * post_build["filter_peak_transmission"]
    )
    ds.attrs["hawcsimulator_instrument"] = definition["name"]
    return CalibrationDatabase(ds)


def row_tangent_altitudes(definition: dict) -> np.ndarray:
    """Tangent altitude of the centre of each line of sight [m]."""
    low, high = definition["viewing"]["tangent_altitude_range_m"]
    edges = np.linspace(low, high, definition["viewing"]["num_rows"] + 1)
    return 0.5 * (edges[1:] + edges[:-1])


def auto_exposure(
    cal_db: CalibrationDatabase,
    wavenumber: np.ndarray,
    radiance: np.ndarray,
    full_well_fraction: float,
) -> float:
    """
    Exposure time [s] that puts the brightest pixel at ``full_well_fraction`` of its
    saturation level: showlib.l1a.simulate.simulate_l1a's automatic exposure for one
    image of ``radiance`` (rows by wavenumber), computed a row at a time.
    """
    ds = cal_db.ds
    gain = float(ds["detector_gain"])
    dark_offset = cal_db.static_dark(0.0)
    dark_current = cal_db.static_dark(1.0) - dark_offset
    max_count = 2 ** int(ds["detector_adc_bits"]) - 1
    capacity = np.minimum(
        float(ds["detector_full_well"]), (max_count - dark_offset) * gain
    )
    widths = bin_width(wavenumber)
    peak = 0.0
    for row, spectrum in enumerate(radiance):
        count_rate = cal_db.forward_operator(row, wavenumber, widths, 1.0) @ spectrum
        peak = max(
            peak, np.max((count_rate + dark_current[row]) * gain / capacity[row])
        )
    return full_well_fraction / peak
