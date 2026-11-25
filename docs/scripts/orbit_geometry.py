"""
The solar geometry along the nominal HAWC orbit, for the docs (docs/source/orbit).

Writes docs/source/img/orbit_geometry_{light,dark}.png and prints the table in
docs/source/orbit/introduction.md. Run from the repository root:

    uv run python docs/scripts/orbit_geometry.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hawcsimulator.geometry.ecef import ECEFLimbGeometry
from hawcsimulator.geometry.orbit import (
    NOMINAL_ORBIT,
    limb_look_vectors,
    time_at_tangent_latitude,
)

mpl.use("Agg")

TANGENT_ALTITUDE = 25e3
# The simulator's default maximum_allowed_sza
MAXIMUM_SZA = 88.0
FIGURE_DATES = {
    "March equinox": "2026-03-20",
    "June solstice": "2026-06-21",
    "December solstice": "2026-12-21",
}
TABLE_DATES = {
    "March equinox": "2026-03-20",
    "June solstice": "2026-06-21",
    "September equinox": "2026-09-22",
    "December solstice": "2026-12-21",
}
# Along the orbit: (latitude, northbound)
TABLE_CROSSINGS = [
    (-75.0, False),
    (-60.0, True),
    (-30.0, True),
    (0.0, True),
    (30.0, True),
    (60.0, True),
    (75.0, False),
]
TICK_CROSSINGS = [
    (-60.0, False),
    (-60.0, True),
    (-30.0, True),
    (0.0, True),
    (30.0, True),
    (60.0, True),
    (60.0, False),
]
# Line labels sit in a column right of the lines, joined to the line ends
LABEL_X = 34.0
# Left edge of the plots [minutes from the northbound equator crossing]
LEFT = -34.0
RIGHT = LABEL_X + 14.0
LABEL_MIN_GAP = 16.0

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#f0efec",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70"],
    },
}


def geometry(time) -> tuple[float, float, float]:
    """Tangent latitude, solar zenith angle and scattering angle at 25 km [degrees]."""
    state = NOMINAL_ORBIT.state(time)
    look = limb_look_vectors(
        state.position[0], state.along_track[0], [TANGENT_ALTITUDE]
    )
    ds = ECEFLimbGeometry(state.time[0], state.position[0], look).geometry_ds
    sza = float(ds["solar_zenith_angle"][0])
    relative_azimuth = float(ds["relative_solar_azimuth_angle"][0])
    # As aliprocessing's L2 defines it: 0 is forward scatter
    scattering = np.degrees(
        np.arccos(np.sin(np.deg2rad(sza)) * np.cos(np.deg2rad(relative_azimuth)))
    )
    return float(ds["tangent_latitude"][0]), sza, float(scattering)


def equator_crossing(date: str) -> pd.Timestamp:
    """When the tangent point crosses the equator northbound, on or after ``date``."""
    return time_at_tangent_latitude(NOMINAL_ORBIT, 0.0, np.datetime64(date))


def orbit_samples(date: str) -> np.ndarray:
    """
    (minutes, latitude, sza, scattering) every 10 s over one orbit, with minutes
    from the tangent point's northbound equator crossing.
    """
    crossing = equator_crossing(date)
    half = 0.5 * NOMINAL_ORBIT.nodal_period
    return np.array(
        [
            (seconds / 60.0, *geometry(crossing + pd.Timedelta(seconds=seconds)))
            for seconds in np.arange(-half, half, 10.0)
        ]
    )


def crossing_label(latitude: float, northbound: bool | None = None) -> str:
    name = (
        "0°" if latitude == 0 else f"{abs(latitude):.0f}°{'N' if latitude > 0 else 'S'}"
    )
    if northbound is None:
        return name
    return name + ("↑" if northbound else "↓")


def orbit_ticks(samples: np.ndarray, date: str) -> list[tuple[float, str]]:
    """Tick positions [minutes] and labels: latitude crossings and the extremes."""
    crossing = equator_crossing(date)
    start = crossing - pd.Timedelta(seconds=0.5 * NOMINAL_ORBIT.nodal_period)
    ticks = []
    for latitude, northbound in TICK_CROSSINGS:
        time = time_at_tangent_latitude(
            NOMINAL_ORBIT, latitude, start, ascending=northbound
        )
        minutes = (time - crossing) / pd.Timedelta(minutes=1)
        ticks.append((minutes, crossing_label(latitude, northbound)))
    minutes, latitude = samples[:, 0], samples[:, 1]
    for i in (np.argmin(latitude), np.argmax(latitude)):
        ticks.append((minutes[i], crossing_label(round(latitude[i]))))
    return sorted(ticks)


def plot(
    samples: dict[str, np.ndarray],
    ticks: list[tuple[float, str]],
    theme: dict,
    path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "text.color": theme["text"],
            "axes.labelcolor": theme["secondary"],
            "xtick.color": theme["muted"],
            "ytick.color": theme["muted"],
            "xtick.labelcolor": theme["secondary"],
            "ytick.labelcolor": theme["secondary"],
        }
    )
    fig, axes = plt.subplots(
        2, 1, figsize=(7.5, 6.2), sharex=True, facecolor=theme["surface"]
    )
    for ax in axes:
        ax.set_facecolor(theme["surface"])
        ax.grid(True, color=theme["grid"], linewidth=0.75)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(theme["axis"])
            ax.spines[side].set_linewidth(0.75)
        ax.tick_params(length=0)

    sza_ax, scattering_ax = axes
    ends = []
    for (name, orbit), colour in zip(samples.items(), theme["series"], strict=False):
        minutes, _, sza, scattering = orbit.T
        daylit = sza <= MAXIMUM_SZA
        for ax, values in ((sza_ax, sza), (scattering_ax, scattering)):
            ax.plot(
                minutes,
                np.where(daylit, values, np.nan),
                color=colour,
                linewidth=1.5,
                solid_capstyle="round",
                solid_joinstyle="round",
                label=name,
            )
        end = np.flatnonzero(daylit)[-1]
        ends.append((scattering[end], minutes[end], name, colour))

    # Each line's end gets a dot, and a label in a column to the right, spaced so
    # the labels don't collide, joined to the end by a leader line
    label_y = -np.inf
    for y, x, name, colour in sorted(ends):
        label_y = max(y, label_y + LABEL_MIN_GAP)
        scattering_ax.plot(
            [x, LABEL_X - 1.0],
            [y, label_y],
            color=theme["muted"],
            linewidth=0.75,
        )
        scattering_ax.plot(
            x,
            y,
            "o",
            markersize=6,
            color=colour,
            markeredgecolor=theme["surface"],
            markeredgewidth=1.5,
        )
        scattering_ax.annotate(
            name,
            (LABEL_X, label_y),
            va="center",
            fontsize=9,
            color=theme["secondary"],
        )

    sza_ax.axhline(MAXIMUM_SZA, color=theme["axis"], linewidth=1.0)
    # Right of every line's end, where the plot is empty
    sza_ax.annotate(
        f"Default limit, {MAXIMUM_SZA:.0f}°",
        (RIGHT, MAXIMUM_SZA),
        xytext=(0, -4),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=9,
        color=theme["muted"],
    )
    sza_ax.set_ylim(0, 95)
    sza_ax.set_yticks([0, 30, 60, 90])
    sza_ax.set_ylabel("Solar zenith angle [°]")
    sza_ax.legend(
        loc="lower left",
        frameon=False,
        fontsize=9,
        labelcolor=theme["secondary"],
        ncols=3,
        bbox_to_anchor=(0.0, 1.0),
        handlelength=1.6,
    )

    scattering_ax.axhline(90, color=theme["axis"], linewidth=1.0)
    for text, y, va in (("Forward scatter", 86, "top"), ("Backscatter", 94, "bottom")):
        scattering_ax.annotate(text, (LEFT, y), va=va, fontsize=9, color=theme["muted"])
    scattering_ax.set_ylim(0, 180)
    scattering_ax.set_yticks([0, 45, 90, 135, 180])
    scattering_ax.set_ylabel("Scattering angle [°]")
    scattering_ax.set_xlim(LEFT, RIGHT)
    scattering_ax.set_xticks([x for x, _ in ticks], [label for _, label in ticks])
    scattering_ax.set_xlabel(
        "Tangent point at 25 km along the orbit (\u2191 northbound, \u2193 southbound)"
    )

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=theme["surface"])
    plt.close(fig)


def main() -> None:
    images = Path(__file__).resolve().parents[1] / "source" / "img"
    samples = {name: orbit_samples(date) for name, date in FIGURE_DATES.items()}
    first = next(iter(FIGURE_DATES))
    ticks = orbit_ticks(samples[first], FIGURE_DATES[first])
    for mode, theme in THEMES.items():
        plot(samples, ticks, theme, images / f"orbit_geometry_{mode}.png")

    labels = [crossing_label(*crossing) for crossing in TABLE_CROSSINGS]
    print("| | " + " | ".join(labels) + " |")
    print("|---" * (len(labels) + 1) + "|")
    for name, date in TABLE_DATES.items():
        cells = []
        for latitude, northbound in TABLE_CROSSINGS:
            time = time_at_tangent_latitude(
                NOMINAL_ORBIT, latitude, np.datetime64(date), ascending=northbound
            )
            _, sza, scattering = geometry(time)
            # Past the simulator's limit there is no scattering angle to give
            cells.append(
                f"{sza:.0f}° / {scattering:.0f}°" if sza <= MAXIMUM_SZA else "\u2013"
            )
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
