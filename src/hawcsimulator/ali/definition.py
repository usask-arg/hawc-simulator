"""ALI instrument definitions as aliprocessing calibration databases."""

from __future__ import annotations

import dataclasses

import numpy as np
import xarray as xr
from aliprocessing.cal_db import EARTH_RADIUS, SimpleInstrument, simulated_cal_db
from aliprocessing.schemas import check
from scipy.special import ndtr

from hawcsimulator.definitions import load_detector, read_noise


def simple_instrument(
    definition: dict, overrides: dict | None = None
) -> SimpleInstrument:
    """
    aliprocessing's SimpleInstrument for an instrument definition.

    The sub-images' tangent-altitude coverage follows from the pixel field of view
    and the boresight tangent altitude. For a definition with a ``pixel_geometry``,
    the layout and field of view are its averages, and build_calibration_database
    replaces the per-pixel geometry. ``overrides`` replace SimpleInstrument fields,
    e.g. ``{"center_wavelengths_nm": (745.0, 1020.0)}``.
    """
    detector = load_detector(definition["detector"]["model"])
    mode = detector["gain_modes"][definition["detector"]["gain_mode"]]
    gain = mode["full_well_e"] / 2 ** detector["adc_bits"]
    dark_current = detector["dark_current_e_per_s"] / gain

    layout = _layout(definition, detector)
    optics = definition["optics"]
    channels = definition["channels"]
    l1b = definition["l1b"]

    polarizer_transmission, polarizer_efficiency = polarizer_response(
        definition, np.asarray(channels["centre_wavelengths_nm"], dtype=float)
    )

    orbit_altitude = definition["orbit_altitude_m"]
    observer_radius = EARTH_RADIUS + orbit_altitude
    depression = np.arccos(
        (EARTH_RADIUS + optics["boresight_tangent_altitude_m"]) / observer_radius
    )
    half_height = layout["rows_per_subimage"] / 2 * layout["vertical_fov_rad"]
    coverage = (
        float(observer_radius * np.cos(depression + half_height) - EARTH_RADIUS),
        float(observer_radius * np.cos(depression - half_height) - EARTH_RADIUS),
    )

    instrument = SimpleInstrument(
        rows_per_subimage=layout["rows_per_subimage"],
        gap_rows=layout["gap_rows"],
        num_columns=layout["columns"],
        horizontal_ifov_deg=float(np.degrees(layout["horizontal_fov_rad"])),
        orbit_altitude=orbit_altitude,
        tangent_altitude_coverage=coverage,
        analyzer_angles_deg=layout["analyzer_angles_deg"],
        # Per-channel values, when the definition gives a polarizer model, replace
        # these in build_calibration_database
        analyzer_efficiency=float(np.mean(polarizer_efficiency)),
        analyzer_transmission=float(np.mean(polarizer_transmission)),
        center_wavelengths_nm=tuple(channels["centre_wavelengths_nm"]),
        bandpass_fwhm_nm=channels.get("filter", channels)["fwhm_nm"],
        # The detector QE, and a filter shape when the definition gives one, are
        # applied in build_calibration_database
        peak_throughput=channels.get("filter", channels).get(
            "peak_transmission", channels.get("filter_peak_transmission", 1.0)
        )
        * optics["transmission"],
        throughput_step_nm=channels["throughput_step_nm"],
        aperture_diameter=float(np.sqrt(4 * layout["aperture_area_m2"] / np.pi)),
        vignetting=layout["vignetting"],
        pixel_response_nonuniformity=optics["pixel_response_nonuniformity"],
        detector_gain=gain,
        detector_read_noise=read_noise(definition),
        detector_full_well=float(mode["full_well_e"]),
        detector_adc_bits=detector["adc_bits"],
        dark_offset=float(definition["detector"]["dark_offset_dn"]),
        dark_offset_spread=float(definition["detector"]["dark_offset_spread_dn"]),
        dark_current=dark_current,
        dark_current_spread=dark_current
        * definition["detector"]["dark_current_spread_fraction"],
        altitude_grid=tuple(l1b["altitude_grid_m"]),
        reference_altitudes=tuple(l1b["reference_altitudes_m"]),
        across_track_half_width_deg=l1b["across_track_half_width_deg"],
        num_across_track_bins=l1b["num_across_track_bins"],
    )
    return dataclasses.replace(instrument, **(overrides or {}))


def build_calibration_database(
    definition: dict, instrument: SimpleInstrument
) -> xr.Dataset:
    """
    The calibration database of ``instrument``, built for ``definition``: aliprocessing's
    simulated calibration with the detector's quantum efficiency curve applied to the
    spectral throughput.
    """
    cal_db = simulated_cal_db(instrument)
    detector = load_detector(definition["detector"]["model"])
    qe = detector["quantum_efficiency"]
    wavelength = cal_db["throughput_wavelength"].to_numpy()
    centres = cal_db["center_wavelength"].to_numpy()
    throughput = cal_db["spectral_throughput"].to_numpy()
    if "filter" in definition["channels"]:
        throughput = definition["optics"]["transmission"] * filter_transmission(
            wavelength, centres[:, np.newaxis], definition["channels"]["filter"]
        )
    # Held at the end values outside the tabulated curve, which only the far wings
    # of the passbands reach
    qe_at = np.interp(wavelength, qe["wavelength_nm"], qe["value"])
    cal_db["spectral_throughput"] = cal_db["spectral_throughput"].copy(
        data=throughput * qe_at
    )
    if "pixel_geometry" in definition:
        _apply_pixel_geometry(cal_db, definition, instrument)
    _apply_polarizers(cal_db, definition, instrument)
    cal_db.attrs["hawcsimulator_instrument"] = definition["name"]
    check(cal_db, "cal_db")
    return cal_db


def filter_transmission(
    wavelength: np.ndarray, centre: np.ndarray, spec: dict
) -> np.ndarray:
    """
    Transmission of a bandpass filter described by its specification: flat at
    ``peak_transmission`` over ``flat_fraction`` of the FWHM, raised-cosine edges
    through half the peak at ±FWHM/2, and ``10**-blocking_od`` out of band.
    """
    fwhm = spec["fwhm_nm"]
    edge = (1 - spec["flat_fraction"]) * fwhm
    inner = fwhm / 2 - edge / 2
    distance = np.abs(np.asarray(wavelength) - centre)
    shape = np.where(
        distance <= inner,
        1.0,
        np.where(
            distance >= inner + edge,
            0.0,
            0.5 * (1 + np.cos(np.pi * (distance - inner) / edge)),
        ),
    )
    floor = 10 ** -spec["blocking_od"]
    return floor + (spec["peak_transmission"] - floor) * shape


def polarizer_response(
    definition: dict, wavelength: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transmission for unpolarized light (Mueller m00) and polarizing efficiency
    (diattenuation) of the polarizers at each wavelength.

    A definition gives either ``transmission`` and ``efficiency``, or a polarizer
    model: its principal transmittance k1 and extinction ratio k1/k2, which is
    ``extinction_ratio`` within ``specified_range_nm`` and
    ``extinction_ratio_outside_range`` beyond it. Then m00 = (k1 + k2) / 2 and the
    efficiency is (k1 - k2) / (k1 + k2).
    """
    polarizers = definition["polarizers"]
    wavelength = np.asarray(wavelength, dtype=float)
    if "principal_transmittance" not in polarizers:
        return (
            np.full(wavelength.shape, float(polarizers["transmission"])),
            np.full(wavelength.shape, float(polarizers["efficiency"])),
        )
    low, high = polarizers["specified_range_nm"]
    ratio = np.where(
        (wavelength >= low) & (wavelength <= high),
        polarizers["extinction_ratio"],
        polarizers["extinction_ratio_outside_range"],
    )
    k1 = float(polarizers["principal_transmittance"])
    k2 = k1 / ratio
    return (k1 + k2) / 2, (k1 - k2) / (k1 + k2)


def _apply_polarizers(
    cal_db: xr.Dataset, definition: dict, instrument: SimpleInstrument
) -> None:
    """Each channel's polarizer transmission and efficiency, from polarizer_response."""
    transmission, efficiency = polarizer_response(
        definition, cal_db["center_wavelength"].to_numpy()
    )
    analyzer = cal_db["pixel_analyzer"].to_numpy()
    used = analyzer >= 0
    two_theta = np.deg2rad(2 * np.array(instrument.analyzer_angles_deg))
    k = np.where(used, analyzer, 0)
    q = np.where(
        used, efficiency[:, np.newaxis, np.newaxis] * np.cos(two_theta)[k], np.nan
    )
    u = np.where(
        used, efficiency[:, np.newaxis, np.newaxis] * np.sin(two_theta)[k], np.nan
    )
    cal_db["q_sensitivity"] = cal_db["q_sensitivity"].copy(data=q)
    cal_db["u_sensitivity"] = cal_db["u_sensitivity"].copy(data=u)
    cal_db["analyzer_transmission"] = cal_db["analyzer_transmission"].copy(
        data=np.broadcast_to(
            transmission[:, np.newaxis], cal_db["analyzer_transmission"].shape
        ).copy()
    )


def _layout(definition: dict, detector: dict) -> dict:
    """The SimpleInstrument layout and field of view of an instrument definition."""
    if "pixel_geometry" not in definition:
        layout = definition["layout"]
        optics = definition["optics"]
        return {
            "rows_per_subimage": layout["rows_per_polarization"],
            "gap_rows": layout["gap_rows"],
            "columns": layout["columns"],
            "vertical_fov_rad": optics["pixel_fov_rad"],
            "horizontal_fov_rad": optics["pixel_fov_rad"],
            "analyzer_angles_deg": tuple(definition["polarizers"]["angles_deg"]),
            "aperture_area_m2": optics["aperture_area_per_polarization_m2"],
            "vignetting": optics["vignetting"],
        }

    geometry = definition["pixel_geometry"]
    sub_images = geometry["sub_images"]
    terms = [tuple(term) for term in geometry["terms"]]
    num_sub_images = len(sub_images)
    rows_per_subimage = max(
        last - first + 1 for first, last in (s["rows"] for s in sub_images)
    )
    gap_rows = (detector["rows"] - num_sub_images * rows_per_subimage) // max(
        num_sub_images - 1, 1
    )
    if (
        num_sub_images * rows_per_subimage + (num_sub_images - 1) * gap_rows
        != detector["rows"]
    ):
        msg = f"{definition['name']}: the sub-images do not tile the {detector['rows']} detector rows"
        raise ValueError(msg)
    # Field angle per row and per column at each sub-image's centre
    vertical = np.mean([s["elevation_deg"][terms.index((1, 0))] for s in sub_images])
    horizontal = np.mean([s["azimuth_deg"][terms.index((0, 1))] for s in sub_images])
    return {
        "rows_per_subimage": rows_per_subimage,
        "gap_rows": gap_rows,
        "columns": detector["columns"],
        "vertical_fov_rad": float(np.deg2rad(vertical / geometry["row_scale"])),
        "horizontal_fov_rad": float(np.deg2rad(horizontal / geometry["column_scale"])),
        "analyzer_angles_deg": tuple(
            float(s["polarizer_angle_deg"]) for s in sub_images
        ),
        "aperture_area_m2": float(
            np.mean([s["effective_aperture_area_m2"] for s in sub_images])
        ),
        "vignetting": 0.0,
    }


def _apply_pixel_geometry(
    cal_db: xr.Dataset, definition: dict, instrument: SimpleInstrument
) -> None:
    """
    Replaces the calibration database's sub-image layout, field angles and etendue
    with the definition's ``pixel_geometry``.
    """
    geometry = definition["pixel_geometry"]
    terms = [tuple(term) for term in geometry["terms"]]
    row_scale, column_scale = geometry["row_scale"], geometry["column_scale"]
    blur_elevation, blur_azimuth = geometry["field_stop_blur_deg"]
    shape = (cal_db.sizes["detector_row"], cal_db.sizes["detector_column"])
    rows, columns = np.meshgrid(
        np.arange(shape[0], dtype=float),
        np.arange(shape[1], dtype=float),
        indexing="ij",
    )
    x = (columns - geometry["column_centre"]) / column_scale

    analyzer = np.full(shape, -1)
    elevation = np.full(shape, np.nan)
    azimuth = np.full(shape, np.nan)
    etendue = np.full(shape, np.nan)
    for k, sub_image in enumerate(geometry["sub_images"]):
        first, last = sub_image["rows"]
        band = slice(first, last + 1)
        r = (rows[band] - sub_image["row_centre"]) / row_scale
        xb = x[band]
        el, el_r, el_x = _polynomial(sub_image["elevation_deg"], terms, r, xb)
        az, az_r, az_x = _polynomial(sub_image["azimuth_deg"], terms, r, xb)
        solid_angle = (
            np.abs(el_r * az_x - el_x * az_r)
            / (row_scale * column_scale)
            * np.deg2rad(1.0) ** 2
        )
        low = np.polynomial.polynomial.polyval(
            xb, sub_image["field_stop_low_elevation_deg"]
        )
        high = np.polynomial.polynomial.polyval(
            xb, sub_image["field_stop_high_elevation_deg"]
        )
        az_low, az_high = sub_image["field_stop_azimuth_deg"]
        lit = (
            ndtr((el - low) / blur_elevation)
            * ndtr((high - el) / blur_elevation)
            * ndtr((az - az_low) / blur_azimuth)
            * ndtr((az_high - az) / blur_azimuth)
        )
        used = lit >= geometry["min_lit_fraction"]
        analyzer[band] = np.where(used, k, -1)
        elevation[band] = np.where(used, el, np.nan)
        azimuth[band] = np.where(used, az, np.nan)
        etendue[band] = np.where(
            used, sub_image["effective_aperture_area_m2"] * solid_angle * lit, np.nan
        )

    rng = np.random.default_rng(0)
    etendue *= 1 + instrument.pixel_response_nonuniformity * rng.standard_normal(shape)

    for name, values in (
        ("pixel_analyzer", analyzer),
        ("pixel_elevation_angle", elevation),
        ("pixel_azimuth_angle", azimuth),
        ("pixel_etendue", etendue),
    ):
        cal_db[name] = cal_db[name].copy(data=values.astype(cal_db[name].dtype))


def _polynomial(
    coefficients: list[float],
    terms: list[tuple[int, int]],
    r: np.ndarray,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """sum(c * r**i * x**j) and its derivatives with respect to r and x."""
    value = np.zeros_like(r)
    d_r = np.zeros_like(r)
    d_x = np.zeros_like(r)
    for c, (i, j) in zip(coefficients, terms, strict=True):
        value += c * r**i * x**j
        if i:
            d_r += c * i * r ** (i - 1) * x**j
        if j:
            d_x += c * j * r**i * x ** (j - 1)
    return value, d_r, d_x
