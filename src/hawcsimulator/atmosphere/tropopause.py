from __future__ import annotations

import numpy as np

# Lapse rate that defines the thermal tropopause [K/km]
_LAPSE_RATE_LIMIT = -2.0


def thermal_tropopause_altitude(altitude: np.ndarray, temperature: np.ndarray) -> float:
    """
    Thermal tropopause altitude: where the lapse rate, searched upwards from 5 km,
    first rises above -2 K/km.

    Ported from showlib.por.l2.merra.get_tropopause_height (itself from the OSIRIS
    operational GetTropopauseHeight.m), which expects altitudes in km.

    Parameters
    ----------
    altitude : np.ndarray
        Increasing altitudes [m], spanning at least 5 to 26 km
    temperature : np.ndarray
        Temperature [K] at each altitude

    Returns
    -------
    float
        Tropopause altitude [m], or NaN if none is found
    """
    altitude_km = np.asarray(altitude, dtype=float) / 1000.0
    temperature = np.asarray(temperature, dtype=float)

    # Forward differences at the ends, central differences elsewhere
    up = np.minimum(np.arange(len(altitude_km)) + 1, len(altitude_km) - 1)
    down = np.maximum(np.arange(len(altitude_km)) - 1, 0)
    lapse_rate = (temperature[up] - temperature[down]) / (
        altitude_km[up] - altitude_km[down]
    )

    return 1000.0 * _fit_tropopause(lapse_rate, altitude_km, _LAPSE_RATE_LIMIT)


def _fit_tropopause(lapse_rate, altitude_km, threshold) -> float:
    # Start above 5 km to skip low altitude temperature inversions, and search
    # no higher than the first level at or above 26 km
    above_5km = np.flatnonzero(altitude_km >= 5)
    above_26km = np.flatnonzero(altitude_km >= 26)
    if len(above_5km) == 0:
        return np.nan
    idx = int(above_5km[0])
    last_idx = int(above_26km[0]) if len(above_26km) else len(altitude_km) - 1

    # If the start is already above the tropopause, step down until below it
    while idx >= 0 and lapse_rate[idx] > threshold:
        idx -= 1
    if idx < 0:
        return np.nan

    alts = altitude_km[idx : last_idx + 1]
    rates = lapse_rate[idx : last_idx + 1]
    above = np.flatnonzero(rates > threshold)
    if len(above) == 0:
        return np.nan

    # Interpolate to where the lapse rate crosses the threshold
    i = above[0]
    x1, x2 = alts[i - 1], alts[i]
    y1, y2 = rates[i - 1], rates[i]
    return float(x1 + (threshold - y1) * (x2 - x1) / (y2 - y1))
