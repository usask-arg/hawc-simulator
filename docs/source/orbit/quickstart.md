(_orbit_quickstart)=
# Quickstart

## Simulating an observation on the orbit
`time_at_tangent_latitude` finds when the tangent point reaches a latitude, and
`config={"observation_method": "orbit"}` takes the viewing geometry from the orbit at that time:

```{code-block} python
import numpy as np

from hawcsimulator.ali import ALISimulator
from hawcsimulator.geometry.observation import observation_geometry
from hawcsimulator.geometry.orbit import NOMINAL_ORBIT, time_at_tangent_latitude

# The first time on or after 21 June 2026 that the tangent point crosses 45 N
# northbound, which is on the daylit side of the orbit
time = time_at_tangent_latitude(NOMINAL_ORBIT, 45.0, np.datetime64("2026-06-21"))

data = ALISimulator().run(
    ["observation", "l2", "l2_truth"],
    {"time": time, "l2_cfg": {"minimizer": {"verbose": 0}}},
    config={"observation_method": "orbit", "atmosphere_method": "omps_calipso_era5"},
)

geometry = observation_geometry(data["observation"].observation)
```

`observation_geometry` gives each line of sight's tangent point, observer position, solar zenith
angle and relative solar azimuth (0° is forward scatter).  At 45° N in June the 25 km tangent point
has a solar zenith angle of about 25° and a relative solar azimuth of about 44°: forward scatter.
`SHOWSimulator` takes the orbit the same way.

## Choosing the time
`time_at_tangent_latitude(orbit, latitude, after)` returns the first time from `after` that the
25 km tangent point crosses `latitude`, moving north, as a UTC `pandas.Timestamp`; `ascending=False`
gives the southbound crossing instead, and `tangent_altitude` follows a different line of sight.  On
the nominal orbit the northbound crossing is in daylight, and the southbound one is dark except near
the summer pole: in June, for example, the tangent point at 75°N is in daylight both ways.  The
simulator rejects an observation whose solar zenith angle at 25 km is over `maximum_allowed_sza`
(88° by default).

Each crossing starts from the previous one, so these times are along one pass:

```{code-block} python
times = []
after = np.datetime64("2026-06-21")
for latitude in (-30.0, 0.0, 30.0, 60.0):
    after = time_at_tangent_latitude(NOMINAL_ORBIT, latitude, after)
    times.append(after)

simulator = ALISimulator()
results = [
    simulator.run(
        ["l2", "l2_truth"],
        {"time": time, "l2_cfg": {"minimizer": {"verbose": 0}}},
        config={"observation_method": "orbit", "atmosphere_method": "omps_calipso_era5"},
    )
    for time in times
]
```

## Looking ahead
`"look_azimuth": 0.0` in the input looks ahead along the orbit, which puts forward scatter in the
southern hemisphere instead.  The tangent point is then ahead of the spacecraft, so give
`time_at_tangent_latitude` the same `look_azimuth`:

```{code-block} python
time = time_at_tangent_latitude(
    NOMINAL_ORBIT, -45.0, np.datetime64("2026-12-21"), look_azimuth=0.0
)
data = ALISimulator().run(
    ["l2"],
    {"time": time, "look_azimuth": 0.0},
    config={"observation_method": "orbit", "atmosphere_method": "omps_calipso_era5"},
)
```

## Other orbits
`orbit` in the input replaces the nominal orbit:

```{code-block} python
from hawcsimulator.geometry.orbit import CircularOrbit

# Sun-synchronous at 450 km, crossing the equator northbound at 10:30 local time
orbit = CircularOrbit.sun_synchronous(450e3, 10.5, np.datetime64("2026-01-01T00:00:00"))

time = time_at_tangent_latitude(orbit, 45.0, np.datetime64("2026-06-21"))
data = ALISimulator().run(
    ["l2"],
    {"time": time, "orbit": orbit},
    config={"observation_method": "orbit", "atmosphere_method": "omps_calipso_era5"},
)
```

`orbit.state(times)` gives the spacecraft's ECEF positions and directions of motion at any times,
and `orbit.local_time_of_ascending_node(times)` the local time it crosses the equator northbound.
See [the HAWC orbit](introduction.md) for the orbit model and its limitations.
