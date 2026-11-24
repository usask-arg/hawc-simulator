(_ali_intro)=
# The Aerosol Limb Imager (ALI) Simulator

`hawcsimulator.ali.ALISimulator` simulates ALI from the atmosphere to the retrieved aerosol
extinction and particle size, using the simple instrument model and processing in
[`aliprocessing`](https://github.com/usask-arg/ali-processing).

```{code-block} python
from hawcsimulator.ali import ALISimulator

data = ALISimulator().run(["l2"], input)
```

## The simulation
Each output below can be requested from `run`:

| Output | Description |
|---|---|
| `atmosphere`, `observation` | The atmosphere and the limb viewing geometry, shared with the SHOW simulator. The lines of sight are ECEF rays from one observer through the tangent point, see [the quickstart](../index.md) |
| `instrument_definition`, `processing_definition`, `acquisition_definition` | The instrument definition, processing version and acquisition pattern, see [Instruments, Processing Versions and Acquisition Patterns](../instruments.md) |
| `ali_instrument` | The instrument definition as an `aliprocessing.cal_db.SimpleInstrument`. For a definition with a per-pixel geometry its layout and field of view are averages, and `calibration_database` replaces the per-pixel geometry |
| `calibration_database` | The instrument model from `aliprocessing.cal_db.simulated_cal_db`, with the definition's filter shapes and polarizers and the detector's quantum efficiency curve |
| `front_end_radiance` | SASKTRAN2 I, Q and U at the centre of each spectral channel, against tangent altitude. The solar irradiance is `aliprocessing.solar.ChannelSolarIrradiance`, the solar spectrum averaged through each channel's spectral response, which replaces the atmosphere's; the radiance is then the band-averaged radiance the instrument model and the L2 retrieval expect, where the atmosphere varies slowly across a passband |
| `pointing` | The spacecraft position (the observation's observer), the boresight azimuth, and for each observation set the boresight elevation and roll, pointed so the boresight tangent point is at the acquisition pattern's boresight tangent altitude |
| `scene` | The front end radiance arranged for the instrument model, by `aliprocessing.forward_model.scene_from_radiance` |
| `l1a` | Detector frames from `aliprocessing.forward_model.simulate_l1a`, with shot, read and quantization noise and saturation. The acquisition pattern sets the frames: each observation set steps through every spectral channel, followed by a blank frame; each frame's exposure, automatic or given; and the frame times, from the exposures, the detector's dead time and the filter-wheel change time. When the instrument definition limits the single exposure, a frame averages equal exposures within the limit |
| `l1b` | Radiance, Stokes parameters and degree of linear polarization on an altitude grid, from `aliprocessing.processing.l1a_to_l1b.process_l1a_to_l1b` |
| `truth` | The true atmospheric state the radiance was calculated from, as an `xarray.Dataset` on the simulation's altitude grid: pressure, temperature, `<gas>_vmr`, aerosol extinction with its wavelength and median radius, surface albedo and tropopause altitude. `truth.to_netcdf(path)` saves it |
| `l2_truth` | The truth on the L2's grid with the L2's names (`aerosol_extinction` when the truth's aerosol is at the L2's wavelength, `aerosol_median_radius`, `surface_albedo`), so `data["l2"] - data["l2_truth"]` is the retrieval error |
| `program_of_record` | Pressure, temperature and tropopause altitude from the simulation atmosphere |
| `l2` | The L2 granule of the chosen observation sets and across-track bins (by default the first set's middle bin), in aliprocessing's L2 format and checked against its schema: `aerosol_extinction` and `aerosol_median_radius` with their noise and solution errors, priors and averaging kernels, `surface_albedo`, the cloud diagnostics and the retrieval's lower and upper altitude limits, along `observation_set`, `across_track` and `altitude`. Each image is retrieved with `aliprocessing.processing.l1b_to_l2.process_l1b_to_l2_image`, by default in two stages: the first fits from 5 to 40 km and screens for clouds, the second fits from above any cloud |

Every pixel sees the front end radiance at its own tangent altitude, so the viewing tangent
altitudes must cover the L1B altitude grid; the default, -3 to 56 km, does.

## Solar geometry
L1A to L1B computes the sun's position from the observation time.  When the observation's solar
angles are given with `tangent_solar_zenith_angle` and `tangent_solar_azimuth_angle`, the simulator
replaces the L1B solar angles with them, so the retrieval sees the same sun as the front end
radiance.  When they are computed from the time, the L1B angles are used as they are.

## Options
`instrument` names the instrument definition (default `ALI-0-FWHEEL-REP1`), `processing`
the processing version (default `v0.1`) and `acquisition` the acquisition pattern (default
`nominal`); see [Instruments, Processing Versions and Acquisition Patterns](../instruments.md).
The instrument definition sets the orbit altitude; to change it for one run, pass
`observer_altitude` in `overrides`.  A run is adjusted further with:

| Input | Description |
|---|---|
| `cal_db_cfg` | Overrides of the `aliprocessing.cal_db.SimpleInstrument` fields derived from the instrument definition, e.g. `{"center_wavelengths_nm": (745.0, 1020.0)}` |
| `acquisition_cfg` | Overrides of the acquisition pattern's `pointing` and `timing`, e.g. `{"timing": {"num_sets": 3}}` or `{"pointing": {"boresight_tangent_altitude_m": 27000.0}}` |
| `l1a_cfg` | `noise` (default `True`) and `seed`, and `exposure_time` and `num_averaged_exposures` to replace the acquisition pattern's exposure |
| `l1b_cfg` | Overrides of the processing version's `process_l1a_to_l1b` arguments, e.g. a `dark` model |
| `l1b_image_cfg` | Which `observation_set` (default 0) and `across_track` bins (default the middle one) to retrieve, each an index, a list or `"all"`, and overrides of the processing version's `aliprocessing.l1b.data.L1bImage.from_l1b` arguments `low_alt`, `high_alt` and `max_dolp_noise` |
| `l2_cfg` | Overrides of the processing version's retrieval settings, which are themselves overrides of aliprocessing's retrieval configuration (`aliprocessing/l2/retrieval_config.yaml`), e.g. `{"minimizer": {"verbose": 0}}` |
| `sk2_kwargs` | `sasktran2.Config` attributes for the front end radiance |

For example `{"l1a_cfg": {"noise": False}}` gives a noise-free simulation.
