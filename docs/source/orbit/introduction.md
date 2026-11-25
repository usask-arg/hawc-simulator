(_orbit_intro)=
# The HAWC Orbit

With `config={"observation_method": "orbit"}` the SHOW and ALI simulators take their viewing
geometry from an orbit: the spacecraft is where the orbit puts it at `time`, and it takes a
vertical limb image looking along or against its direction of motion.  Without it, the geometry is
placed by hand from a tangent point, see {ref}`Observation geometry <observation-geometry>`.

```{code-block} python
data = ALISimulator().run(
    ["l2"], {"time": time}, config={"observation_method": "orbit"}
)
```

## The nominal orbit
`hawcsimulator.geometry.orbit.NOMINAL_ORBIT` is the default `orbit`:

| Parameter | Nominal orbit |
|---|---|
| Altitude | 450 km above the equatorial radius; 450 to 471 km above the WGS84 ellipsoid |
| Inclination | 97.21°, sun-synchronous |
| Local time of the ascending node | 13:30, mean solar time |
| Nodal period | 93.71 minutes, 15.37 orbits a day |
| Epoch | Ascending node at 2026-01-01 00:00 UTC, at 157.5° W |

The orbit crosses the equator northbound at 13:30 local time and southbound at 01:30 on every orbit,
so the tangent point is mostly in daylight on the northbound half of each orbit and dark on the
southbound half.  Near the summer pole the daylight carries on past the most northerly or southerly
point: in June the tangent point stays in daylight on the southbound half down to about 65°N, and in
December it is already in daylight on the southbound half from about 69°S.  The epoch only sets where
the spacecraft is at a given time; `time_at_tangent_latitude` finds the time of an observation, see
[the quickstart](quickstart.md).

## Looking behind the spacecraft
The lines of sight lie in the vertical plane at `look_azimuth` from the spacecraft's direction of
motion, measured clockwise looking down: 0 looks ahead along the orbit and 180 behind it.  The
default, 180, looks behind.  On the northbound half of the orbit the instrument then looks south,
towards the sun in the northern hemisphere and away from it in the southern hemisphere.  On the
daylit stretches of the southbound half, near the summer pole, it looks north, towards the midnight
sun in the Arctic and away from it in the Antarctic.  Either way it sees forward scatter in the north
and backscatter in the south, in every season.  The 25 km tangent point is about 2,400 km behind the
spacecraft, 20.5° of arc, where the spacecraft was 5.4 minutes earlier.

```{image} ../img/orbit_geometry_light.png
:class: only-light
:alt: Solar zenith angle and scattering angle along the orbit for the March equinox and the June and December solstices
```
```{image} ../img/orbit_geometry_dark.png
:class: only-dark
:alt: Solar zenith angle and scattering angle along the orbit for the March equinox and the June and December solstices
```

The solar zenith angle and scattering angle at the 25 km tangent point along one orbit, against the
time from the tangent point's northbound equator crossing, labelled by the tangent point's latitude
and direction (↑ northbound, ↓ southbound).  The scattering angle is as aliprocessing's L2 reports
it, $\arccos(\sin\theta_0 \cos\phi)$ for solar zenith angle $\theta_0$ and relative solar azimuth
$\phi$: 0° is forward scatter and 180° backscatter.  The lines cover the daylit part of the orbit,
where the solar zenith angle is at most 88°, the simulator's default `maximum_allowed_sza`.  The same
values at a few points along the orbit, solar zenith angle / scattering angle:

| | 75°S↓ | 60°S↑ | 30°S↑ | 0°↑ | 30°N↑ | 60°N↑ | 75°N↓ |
|---|---|---|---|---|---|---|---|
| March equinox | – | 65° / 147° | 38° / 120° | 21° / 92° | 34° / 64° | 61° / 37° | – |
| June solstice | – | – | 59° / 144° | 32° / 116° | 17° / 87° | 37° / 58° | 77° / 22° |
| September equinox | – | 67° / 145° | 41° / 121° | 24° / 93° | 35° / 66° | 60° / 40° | – |
| December solstice | 82° / 154° | 44° / 125° | 25° / 97° | 32° / 70° | 56° / 43° | 84° / 25° | – |

– : the solar zenith angle is over 88°.  `docs/scripts/orbit_geometry.py` makes the figure and the
table.

## The orbit model
`CircularOrbit` is a circular orbit propagated analytically: its node and argument of latitude drift
at the secular rates the Earth's oblateness (J2) causes, which is what keeps a sun-synchronous orbit
at the same local time.  The Earth rotates at the sidereal rate about its axis; precession,
nutation, polar motion, drag and eccentricity are left out.  That places observations realistically
in latitude, season and local solar time, but it is not an orbit determination.  It agrees with the
example HAWCsat orbit in the SASKTRAN2 database, which another tool made, to a few tens of km over an
orbit.

Other orbits:

- `CircularOrbit.sun_synchronous(altitude, local_time_of_ascending_node, ascending_node_time)`, a
  sun-synchronous orbit with its inclination from the altitude
- `CircularOrbit(altitude, inclination, ascending_node_time, ascending_node_longitude)`, any circular
  orbit

## How an observation is built
`observation__orbit`

1. finds the spacecraft's position and direction of motion at `time`, with `CircularOrbit.state`
2. makes one line of sight per viewing tangent altitude, in the vertical plane at `look_azimuth`, each
   reaching its tangent altitude above the WGS84 ellipsoid, with `limb_look_vectors`
3. gives each line of sight to SASKTRAN2 as an ECEF ray (`ECEFLimbGeometry`), with the solar angles
   at its own tangent point from the sun's position at `time` (`SunDirectionSolarHandler`)

The rest of the simulation is the same as for the hand-placed geometry: ALI points its boresight
along these lines of sight, and the radiance is calculated in a one-dimensional atmosphere with the
solar zenith angle at the 25 km tangent point.  `observation_geometry(data["observation"].observation)`
gives each line of sight's geometry.

## Limitations
- The spacecraft stays where it is at `time` for the whole observation.  An ALI observation set takes
  a few seconds, in which the tangent point moves about 7 km/s along the orbit, but all its L1A frames
  are taken from the same position.
- The atmosphere does not change along the orbit or along the lines of sight.
- The instrument definitions assume a 450 km orbit (`orbit_altitude_m`), which sizes their fields of
  view.  The nominal orbit is 450 to 471 km above the ellipsoid; another orbit altitude changes the
  altitudes the instruments cover.

The geometry keeps each line of sight's time, observer position and look direction, which is what
SASKTRAN2's orbital-plane engine takes for along-track atmospheres.
