from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import appdirs
import numpy as np
import xarray as xr


def _default_filter(w, wl):
    f = np.ones(w.shape)
    f[w < wl] = 0.0

    return f


def bandpass_filter(frequencies, c, w, delta_w, A_stop):
    """
    Compute the magnitude response of a bandpass filter at specified frequencies.

    Parameters:
    frequencies : array-like
        Frequencies at which to compute the filter response.
    c : float
        Center frequency of the bandpass filter.
    w : float
        Half-width of the passband (i.e., passband is from c-w to c+w).
    delta_w : float
        Width of the transition bands on each side.
    A_stop : float
        Stopband attenuation in dB (positive value).

    Returns:
    H : array-like
        Magnitude response of the filter at the specified frequencies.
    """
    frequencies = np.asarray(frequencies)
    H = np.zeros_like(frequencies, dtype=float)

    H_stop = 10 ** (-A_stop / 20.0)  # Stopband gain (linear scale)

    # Passband indices
    idx_passband = np.logical_and(frequencies >= c - w, frequencies <= c + w)
    H[idx_passband] = 1.0

    # Lower transition band
    idx_lower_transition = np.logical_and(
        frequencies >= c - w - delta_w, frequencies < c - w
    )
    H[idx_lower_transition] = H_stop + 0.5 * (1.0 - H_stop) * (
        1 + np.cos(np.pi * (frequencies[idx_lower_transition] - (c - w)) / delta_w)
    )

    # Upper transition band
    idx_upper_transition = np.logical_and(
        frequencies > c + w, frequencies <= c + w + delta_w
    )
    H[idx_upper_transition] = H_stop + 0.5 * (1.0 - H_stop) * (
        1 + np.cos(np.pi * (frequencies[idx_upper_transition] - (c + w)) / delta_w)
    )

    # Stopband indices (outside passband and transition bands)
    idx_stopband_lower = frequencies < c - w - delta_w
    idx_stopband_upper = frequencies > c + w + delta_w
    H[idx_stopband_lower] = H_stop
    H[idx_stopband_upper] = H_stop

    return H


def generate_ideal_l2_cal_db(
    num_samples: int,
    opd_per_sample: float,
    littrow_wavel_nm: float,
    apodization: float | None = None,
    filter_fn: Callable = _default_filter,
    hires_reduction_factor: int = 10,
    fc=0.01,
    include_aliasing=False,
) -> xr.Dataset:
    result = xr.Dataset()

    littrow_wavenumber = 1e7 / littrow_wavel_nm

    wvnum_spacing = 1 / (2 * num_samples * opd_per_sample)

    # Only include positive samples
    delta_wvnum = np.arange(0, num_samples // 2) * wvnum_spacing

    # Generate the wavenumber grid
    sample_wv = littrow_wavenumber - delta_wvnum
    good_samples = filter_fn(1e7 / sample_wv, littrow_wavel_nm) > fc
    result["sample_wavenumber"] = sample_wv[good_samples]

    hires_wavenumber_grid = np.arange(
        littrow_wavenumber - delta_wvnum[-1],
        littrow_wavenumber + delta_wvnum[-1],
        wvnum_spacing / hires_reduction_factor,
    )

    result["hires_wavenumber"] = hires_wavenumber_grid
    result["filter"] = (
        ["hires_wavenumber"],
        filter_fn(hires_wavenumber_grid, littrow_wavel_nm),
    )

    ils = np.zeros((len(hires_wavenumber_grid), len(result.sample_wavenumber)))

    for idx, wvnum in enumerate(result.sample_wavenumber):
        ideal_ils = np.sinc((hires_wavenumber_grid - float(wvnum)) / (wvnum_spacing))

        if apodization is not None:
            apodized_ils = np.fft.fftshift(np.fft.fft(np.fft.fftshift(ideal_ils)))
            apod_fun = np.zeros(len(apodized_ils))
            c = len(apod_fun) // 2
            apod_fun[c - (num_samples // 2) : (c + num_samples // 2)] = np.kaiser(
                num_samples, apodization
            )

            ideal_ils = np.fft.fftshift(
                np.fft.ifft(np.fft.ifftshift(apod_fun * apodized_ils)).real
            )

        if include_aliasing:
            # Have to add on the aliased part of the ILS
            ideal_ils += ideal_ils[::-1]

        ideal_ils *= filter_fn(1e7 / hires_wavenumber_grid, littrow_wavel_nm)

        ils[:, idx] = ideal_ils

    result["ils"] = (["hires_wavenumber", "sample_wavenumber"], ils)

    return result


def calibration_database(name: str = "ideal", version: str = "v1"):
    dir = (
        Path(
            appdirs.AppDirs(
                appname="hawc-simulator", appauthor="usask-arg"
            ).user_data_dir
        )
        / "show"
        / "calibration"
    )

    file = dir / f"{name}_{version}.nc"

    if not file.exists():
        if name == "ideal":
            cal_db = generate_ideal_l2_cal_db(
                512,
                0.002 * 3.4,
                1362,
                apodization=10,
                include_aliasing=False,
                filter_fn=lambda w, wl: bandpass_filter(  # noqa: ARG005
                    w, 1364, 1.0, 0.2, 80
                ),
            )
            file.parent.mkdir(parents=True, exist_ok=True)
            cal_db.to_netcdf(file)

            return file
        error_message = f"Unknown calibration database name: {name}"
        raise ValueError(error_message)
    return file
