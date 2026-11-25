(_orbit_sampling)=
# Sampling Locations

The `sampling` output of the SHOW and ALI simulators lists when and where the instrument measures
over a period: the time of every observation and the geometry of its tangent point, from the orbit
and the acquisition pattern.  No radiance is calculated, so a day of sampling takes a second or two.

```{code-block} python
import pandas as pd

from hawcsimulator.ali import ALISimulator

sampling = ALISimulator().run(
    ["sampling"],
    {
        "sampling_start": pd.Timestamp("2026-06-21T00:00:00Z"),
        "sampling_end": pd.Timestamp("2026-06-22T00:00:00Z"),
    },
)["sampling"]
```

`sampling` is an `xarray.Dataset` along `time`, the start of each observation.  ALI is an imager,
so each of its observations has a profile in every L1B across-track bin, five for
`ALI-0-FWHEEL-REP1`, along `across_track`, with `across_track_angle` the bin's centre [degrees,
positive to the right looking along the boresight].  SHOW has one profile per image.

| Variable | Description |
|---|---|
| `tangent_latitude`, `tangent_longitude` | The tangent point at 25 km [degrees]; ALI's along `across_track` |
| `solar_zenith_angle` | At the tangent point [degrees]; ALI's along `across_track` |
| `relative_solar_azimuth_angle` | The sun's azimuth from the line of sight's at the tangent point [degrees, −180 to 180, 0 is forward scatter]; ALI's along `across_track` |
| `scattering_angle` | $\arccos(\sin\theta_0 \cos\phi)$, as aliprocessing's L2 reports it [degrees, 0 is forward scatter]; ALI's along `across_track` |
| `los_azimuth_angle` | The line of sight's azimuth at the tangent point [degrees clockwise from north]; ALI's along `across_track` |
| `spacecraft_latitude`, `spacecraft_longitude`, `spacecraft_altitude` | The spacecraft's position [degrees, m] |
| `ascending` | Whether the tangent point is moving north |

Its attributes record the instrument definition, processing version and acquisition pattern, as for
the other outputs, and the sampling interval.  Each observation has exactly the geometry an
[orbit observation](quickstart.md) at that time has at 25 km, and ALI's across-track profiles are
where its L1B places those bins, so `sampling` can pick the times to simulate:

```{code-block} python
simulator = ALISimulator()

# The middle across-track bin, which the L2 retrieves by default
middle = sampling.isel(across_track=sampling.sizes["across_track"] // 2)
northern = middle.isel(time=middle["tangent_latitude"] > 60)

data = simulator.run(
    ["l2", "l2_truth"],
    {"time": pd.Timestamp(northern["time"].to_numpy()[0], tz="UTC")},
    config={"observation_method": "orbit", "atmosphere_method": "omps_calipso_era5"},
)
```

`"l1b_image_cfg": {"across_track": "all"}` in the input retrieves every across-track bin.

## When the instruments measure
Observations start every `sampling.interval_s` of the acquisition pattern from `sampling_start`,
and those whose tangent point at 25 km is in daylight, with a solar zenith angle of at most
`maximum_allowed_sza` (88° by default), are kept.  The nominal acquisition patterns' intervals:

| Instrument | Interval | |
|---|---|---|
| SHOW | 2.4 s | Back to back: each image integrates 2 s as 22 exposures, each followed by the detector's dead time, 2.37 s in all |
| ALI | 20 s | With automatic exposure an observation set takes about 10 to 18 s, depending on the scene's brightness |

`acquisition_cfg` changes the interval for a run, for example
`{"sampling": {"interval_s": 30.0}}`.  When the acquisition pattern's timing is fixed, an interval
shorter than one observation is an error; with automatic exposure the length of an observation
depends on the scene, so it is not checked.  `orbit`, `look_azimuth` and `maximum_allowed_sza` in the
input work as for an orbit observation.

ALI's across-track profiles are the image's columns at the bin centres, as aliprocessing's camera
model has them: the boresight points at the acquisition pattern's `pointing.boresight_tangent_altitude_m`
and is rolled by `pointing.boresight_roll_deg`, and each column's line of sight reaches 25 km.
With the nominal pattern the five profiles span about 170 km across the track, ±2° at about 2,400 km.

## A day of sampling

```{image} ../img/hawc_sampling_light.png
:class: only-light
:alt: Maps of a day of ALI tangent points in each season, coloured by scattering angle
```
```{image} ../img/hawc_sampling_dark.png
:class: only-dark
:alt: Maps of a day of ALI tangent points in each season, coloured by scattering angle
```

A day of ALI profiles, an observation every 20 s on the nominal orbit with five across-track
profiles each, in each season, coloured by the scattering angle at the 25 km tangent point.  Each
track is the daylit part of one orbit, about 15 a day, running from south to north and moving west
from one orbit to the next; its width is the 170 km across-track swath.  Near the summer pole the
track hooks back: past the most northerly point, at about 83°N, the tangent point is still in
daylight on its way south, and the same happens past the most southerly point in December.  SHOW
samples the middle of the same tracks, every 2.4 s.

| | ALI observations | ALI profiles | SHOW images | Tangent latitudes | Forward scatter |
|---|---|---|---|---|---|
| March equinox | 2,093 | 10,465 | 17,449 | 82°S to 84°N | 49% |
| June solstice | 2,096 | 10,480 | 17,478 | 60°S to 84°N | 49% |
| September equinox | 2,134 | 10,670 | 17,789 | 82°S to 84°N | 49% |
| December solstice | 2,148 | 10,740 | 17,902 | 84°S to 65°N | 49% |

Observations and profiles in a day, the latitudes ALI's profiles cover, and the share of them in
forward scatter (scattering angle under 90°).  `docs/scripts/hawc_sampling.py` makes the figure and
the table.
