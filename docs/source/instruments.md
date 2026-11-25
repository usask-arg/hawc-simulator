(_instruments)=
# Instruments, Processing Versions and Acquisition Patterns

The simulators describe each instrument design by a named **instrument definition**, the
L1A to L1B and L2 algorithm settings by a **processing version**, and how the instrument is
pointed and exposed by an **acquisition pattern**.  The three are versioned separately and
chosen with the `instrument`, `processing` and `acquisition` inputs:

```{code-block} python
from hawcsimulator.show import SHOWSimulator

data = SHOWSimulator().run(
    ["l2"],
    {**geometry, "instrument": "SHOW-0-SHS-REP1", "processing": "v0.1", "acquisition": "nominal"},
)
```

Each defaults to the current recommended choice, listed below.  Every product the simulators
write (`l1a`, `l1b`, `l2`) records them and the versions of hawcsimulator, showlib or
aliprocessing, skretrieval and sasktran2 in its attributes (`hawcsimulator_instrument`,
`hawcsimulator_processing`, `hawcsimulator_acquisition`, `..._version`), and the `provenance`
output holds the same.

## Instrument names
Instrument definitions are named `INSTRUMENT-PHASE-CONCEPT-VERSION`:

| Field | Meaning | Examples |
|---|---|---|
| INSTRUMENT | The instrument | `SHOW`, `ALI` |
| PHASE | The mission phase of the design | `0`, `A`, `B` |
| CONCEPT | The design concept | `SHS`, `FWHEEL` |
| VERSION | A label with a revision number | `REP1` (representative), `PRELIM1`, `BASELINE1`, `FM1` |

Names are uppercase letters and digits with `-` only between fields.  A published name is
never changed: a corrected or updated definition gets a new revision, e.g. `REP2`.

Processing versions are named `v0.X` until launch, when `v1.0` is defined.

## Available instruments
Both are **representative** phase 0 instruments (`REP`): they have the phase 0 designs'
coverage, sampling and spectral bands, and performance close to the designs', but are
described by performance-level parameters rather than by the designs themselves.  The
designs are in `hawcsimulator-internal` (see [Internal definitions](#internal-definitions)).

Both use a Raptor Owl 1280 detector in low gain.  The detector values (12-bit ADC; 450 ke⁻
full well; 160 e⁻ read noise; single exposures of 20 µs to 92.5 ms; dark current 19,000 e⁻/s,
the specification's upper limit at 15 °C, the assumed operating temperature) come from the
public Owl 1280 datasheet.  Integrations longer than 92.5 ms average equal exposures, each
adding its own read noise.  The dead time between exposures (16.7 ms, one 60 Hz full-frame
readout) is assumed and the dark offset is a placeholder; an instrument definition sets the
read noise and dead time with `detector.read_noise_e` and `detector.dead_time_s`.

### SHOW-0-SHS-REP1 (default)
Spatial heterodyne spectrometer.

| Parameter | Value |
|---|---|
| Orbit altitude | 450 km |
| Lines of sight | 200, from 0 to 45 km (each a bin of 3.2 detector rows) |
| Across-track field of view | 1.5° |
| Spectral | Littrow wavenumber 7345 cm⁻¹, resolution 0.13 cm⁻¹, 1024 interferogram samples with zero path difference at the centre |
| Fringe visibility | 0.8 (assumed interferometer modulation efficiency) |
| Throughput | 83 cm² effective area: the collecting area times the optics, interferometer and detector efficiencies |
| Filter | 1369.7 nm centre, 3.8 nm FWHM, 75% peak, 40 dB blocking, raised-cosine edges |

### ALI-0-FWHEEL-REP1 (default)
Filter-wheel imager.

| Parameter | Value |
|---|---|
| Orbit altitude | 450 km |
| Layout | three polarization sub-images of 328 rows, 20 rows apart; 1280 columns |
| Pixel field of view | 7.1 × 10⁻⁵ rad, the same for every pixel and without distortion; each sub-image covers about −3 to 52 km and 5.2° across track, centred on the boresight |
| Etendue | 1.25 cm² effective aperture times the pixel solid angle, the same for every pixel (1% response nonuniformity) |
| Channels | 525, 745, 869, 1020, 1560 nm; 5 nm FWHM filters, flat at 85% over the central 80% of the band with raised-cosine edges, OD 4.5 blocking out of band (centre-wavelength error and ripple not modelled) |
| Polarizers | Thorlabs LPVIS nanoparticle film, specified for 550–1500 nm, at 60°, 0° and −60° (stand-ins): extinction ratio 10,000:1 in range (listed) and 100:1 outside it (assumed, the 525 and 1560 nm channels); principal transmittance 0.85 (assumed) |
| Optics transmission | 0.8 (assumed) |
| Detector QE | the Owl 1280 datasheet curve |
| L1B | 0–50 km in 250 m steps, 5 across-track bins over ±2.5° |

The filters' blocking applies across the whole spectral throughput table, about 500–1585 nm,
where it is 0.6–2% of each channel's response; the front end radiance is computed at the
channel centres, so out-of-band light from other wavelengths is not simulated.

## Processing versions

| Version | SHOW | ALI |
|---|---|---|
| v0.1 (default) | showlib's L1A to L1B (Kaiser apodization β = 10) and L2 defaults, with the preprocessor sample chosen automatically, lines of sight up to 35 km used (showlib's default is 30 km), the surface albedo held at its prior value with `prior_influence` 10⁴ (showlib's default, 0, leaves it free), and the minimizer stopping when the cost changes by less than 0.1% (`ftol` 10⁻³; showlib's default is 10⁻⁶) | aliprocessing's L1A to L1B and two-stage L2 defaults |

A processing version holds overrides of the library's defaults, so it is defined together with
the library versions recorded in the outputs.

## Acquisition patterns
An acquisition pattern sets how the instrument is pointed and exposed.  `acquisition_cfg`
adjusts one for a run, for example `{"timing": {"num_sets": 3}}`.

| Instrument | Pattern | Description |
|---|---|---|
| SHOW | `nominal` (default) | each image integrates 2 s (`timing.integration_time_s`; `"auto"` fills the brightest pixel to `timing.full_well_fraction` of full well); images every 2.4 s, back to back (`sampling.interval_s`) |
| ALI | `nominal` (default) | limb stare with the boresight at 25 km; one observation set through every channel, each frame exposed to fill the brightest pixel to 60% of saturation, then a blank frame; observations every 20 s (`sampling.interval_s`) |

An ALI pattern has two sections:
- `pointing`: `boresight_tangent_altitude_m` and `boresight_roll_deg`, the same for every
  observation set.
- `timing`: `num_sets`; `exposure`, the integration per frame in seconds (one value, or one per
  channel) or `"auto"` with `fill`; `filter_change_s`, the time to move the filter wheel; and
  `blank_between_sets`.

Integrations longer than the detector's longest single exposure are averaged from equal
exposures.  Each frame takes its exposures plus the instrument's dead time after each, and the
next frame starts `filter_change_s` later, which sets the L1A frame times.

Both instruments' patterns have a `sampling` section: `interval_s`, the time between the starts of
consecutive observations, which sets the [sampling locations](orbit/sampling.md).  When the timing
is fixed, an interval shorter than one observation (the `observation_duration` output) is an error.

(internal-definitions)=
## Internal definitions
Every definition shipped with hawcsimulator is public and holds only specification-level,
representative or derived values.  Definitions built from non-public information, such as the
instrument designs, live in a separate package,
`hawcsimulator-internal`, available to the team.  Installing it adds its definitions, which are
then used by name like the public ones:

```{code-block} python
from hawcsimulator.definitions import available

available("hawcsimulator.show", "instruments")   # public and installed internal names
```

Any package can provide definitions this way: it declares an entry point in the
`hawcsimulator.definitions` group naming a package laid out like hawcsimulator's
(`show/instruments`, `show/processing`, `show/acquisition`, the same for `ali`, and `detectors`).
A name defined by more than one package is an error.  The outputs record where each definition
came from in `hawcsimulator_instrument_source`, `hawcsimulator_processing_source` and
`hawcsimulator_acquisition_source`, so results made with internal definitions can be recognized.

## Adjusting and adding definitions
For experiments, `cal_db_cfg`, `acquisition_cfg`, `l1b_cfg` and `l2_cfg` adjust a run on top of
the instrument definition, acquisition pattern and processing version; see
[the SHOW simulator](show/introduction.md) and [the ALI simulator](ali/introduction.md).

The definitions are YAML files in `hawcsimulator/show/instruments`,
`hawcsimulator/ali/instruments`, `hawcsimulator/*/processing` and `hawcsimulator/*/acquisition`.
To try a new one, copy a file, give it a new name, and pass its path as the `instrument`,
`processing` or `acquisition` input; to add it to hawcsimulator, add the file to the package.  A simulator with different defaults is a
two-line subclass:

```{code-block} python
class SHOWPhaseASimulator(SHOWSimulator):
    default_instrument = "SHOW-A-SHS-BASELINE1"
```
