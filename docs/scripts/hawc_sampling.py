"""
A day of HAWC sampling locations in each season, for the docs (docs/source/orbit).

Writes docs/source/img/hawc_sampling_{light,dark}.png and prints the table in
docs/source/orbit/sampling.md. Run from the repository root:

    uv run python docs/scripts/hawc_sampling.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from hawcsimulator.ali import ALISimulator
from hawcsimulator.show import SHOWSimulator

mpl.use("Agg")

DATES = {
    "March equinox": "2026-03-20",
    "June solstice": "2026-06-21",
    "September equinox": "2026-09-22",
    "December solstice": "2026-12-21",
}

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        # Diverging: blue (forward scatter) to red (backscatter) through a neutral
        # gray at 90 degrees, mid-lightness so every point shows on the surface
        "diverging": ["#2a78d6", "#898781", "#e34948"],
    },
    "dark": {
        "surface": "#1a1a19",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "diverging": ["#3987e5", "#898781", "#e66767"],
    },
}


def day_of_sampling(simulator, date: str):
    start = pd.Timestamp(date, tz="UTC")
    return simulator.run(
        ["sampling"],
        {"sampling_start": start, "sampling_end": start + pd.Timedelta(days=1)},
    )["sampling"]


def plot(samples: dict, theme: dict, path: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelcolor": theme["secondary"],
            "xtick.labelcolor": theme["secondary"],
            "ytick.labelcolor": theme["secondary"],
        }
    )
    colours = LinearSegmentedColormap.from_list("scattering", theme["diverging"])
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.5, 5.0),
        sharex=True,
        sharey=True,
        facecolor=theme["surface"],
        gridspec_kw={"hspace": 0.28, "wspace": 0.08},
    )
    for ax, (name, sampling) in zip(axes.flat, samples.items(), strict=True):
        ax.set_facecolor(theme["surface"])
        ax.grid(True, color=theme["grid"], linewidth=0.75)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(theme["axis"])
            spine.set_linewidth(0.75)
        ax.tick_params(length=0)
        # Every profile: each observation's across-track bins
        points = ax.scatter(
            sampling["tangent_longitude"].to_numpy().ravel(),
            sampling["tangent_latitude"].to_numpy().ravel(),
            c=sampling["scattering_angle"].to_numpy().ravel(),
            cmap=colours,
            vmin=0.0,
            vmax=180.0,
            s=2,
            linewidths=0,
        )
        ax.set_title(name, fontsize=10, color=theme["secondary"], loc="left")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_aspect("equal")
        # Unlabelled at 180, where neighbouring panels' labels would meet
        ax.set_xticks([-120, -60, 0, 60, 120])
        ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
        ax.xaxis.set_major_formatter(
            lambda v, _: "0°" if v == 0 else f"{abs(v):.0f}°{'E' if v > 0 else 'W'}"
        )
        ax.yaxis.set_major_formatter(
            lambda v, _: "0°" if v == 0 else f"{abs(v):.0f}°{'N' if v > 0 else 'S'}"
        )
        ax.tick_params(labelsize=8)

    bar = fig.colorbar(
        points, ax=axes, orientation="horizontal", fraction=0.05, pad=0.09, aspect=40
    )
    bar.outline.set_visible(False)
    bar.set_ticks([0, 45, 90, 135, 180])
    bar.ax.tick_params(length=0, labelsize=8, labelcolor=theme["secondary"])
    bar.set_label(
        "Scattering angle at the 25 km tangent point [°]: "
        "forward scatter below 90°, backscatter above",
        color=theme["secondary"],
        fontsize=9,
    )
    fig.savefig(path, dpi=150, facecolor=theme["surface"], bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    images = Path(__file__).resolve().parents[1] / "source" / "img"
    ali, show = ALISimulator(), SHOWSimulator()
    samples = {name: day_of_sampling(ali, date) for name, date in DATES.items()}
    for mode, theme in THEMES.items():
        plot(samples, theme, images / f"hawc_sampling_{mode}.png")

    print(
        "| | ALI observations | ALI profiles | SHOW images | Tangent latitudes "
        "| Forward scatter |"
    )
    print("|---|---|---|---|---|---|")
    for name, date in DATES.items():
        sampling = samples[name]
        num_show = day_of_sampling(show, date).sizes["time"]
        latitude = sampling["tangent_latitude"]
        forward = float((sampling["scattering_angle"] < 90).mean())
        print(
            f"| {name} | {sampling.sizes['time']:,} | "
            f"{sampling['tangent_latitude'].size:,} | {num_show:,} | "
            f"{abs(float(latitude.min())):.0f}°S to {float(latitude.max()):.0f}°N | "
            f"{forward:.0%} |"
        )


if __name__ == "__main__":
    main()
