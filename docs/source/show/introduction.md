(_show_introduction)=
# The Spatial Heterodyne Observations of Water (SHOW) Simulator

`hawcsimulator.show.SHOWSimulator` simulates a SHOW limb image from the atmosphere to the retrieved
water vapour profile, using the instrument model and processing in
[`showlib`](https://github.com/usask-arg/show-lib).

```{code-block} python
from hawcsimulator.show import SHOWSimulator

data = SHOWSimulator().run(["l2"], input)
```

## The simulation
Each output below can be requested from `run`:

| Output | Description |
|---|---|
| `atmosphere`, `observation` | The atmosphere and the limb viewing geometry, shared with the ALI simulator. The lines of sight are ECEF rays from one observer through the tangent point, see [the quickstart](../index.md) |
| `instrument_definition`, `processing_definition`, `acquisition_definition` | The instrument definition, processing version and acquisition pattern, see [Instruments, Processing Versions and Acquisition Patterns](../instruments.md) |
| `calibration_database` | The instrument model: the instrument definition as a showlib calibration database (built with `showlib.cal_db.ideal.ideal_calibration_database`), with one detector row per line of sight |
| `front_end_radiance` | SASKTRAN2 radiance (two-stream multiple scattering, every CPU core) on the calibration database's model wavenumber grid. MIPAS CH4 is added to the atmosphere unless it already has `ch4`, since the retrieval models it |
| `l1a` | Interferograms from `showlib.l1a.simulate.simulate_l1a`: the forward model, the dark signal, shot and read noise, saturation and digitization. The image integrates the acquisition pattern's time (2 s by default); when the instrument definition limits the single exposure, the image averages equal exposures within the limit |
| `l1b` | Calibrated radiance spectra from `showlib.processing.l1a_to_l1b.process_l1a_to_l1b` |
| `truth` | The true atmospheric state the radiance was calculated from, as an `xarray.Dataset` on the simulation's altitude grid: pressure, temperature, `<gas>_vmr` (including the CH4 added for the retrieval), aerosol extinction with its wavelength, surface albedo and tropopause altitude. `truth.to_netcdf(path)` saves it |
| `l2_truth` | The truth on the L2's grid with the L2's names (`h2o_vmr`, and `aerosol_extinction` when the truth's aerosol is at the L2's wavelength), so `data["l2"].ds - data["l2_truth"]` is the retrieval error |
| `program_of_record` | Pressure, temperature and tropopause altitude from the simulation atmosphere, and for now, when the atmosphere has aerosol, the true aerosol extinction at 1350 nm (`por_aerosol_wavelength_nm`), which the retrieval starts from and uses as its aerosol prior: showlib's pre-fit aerosol fails when the atmosphere's H2O differs from the climatology it holds H2O at. Set `por_aerosol_wavelength_nm` to `None` to use the pre-fit's aerosol |
| `l2` | The L2 granule from `showlib.processing.l1b_to_l2.retrieve_l2`, a `showlib.l2.data.L2DataSet` checked against showlib's L2 schema. Its `ds` holds, along `time` and `altitude`, `h2o_vmr` with `h2o_vmr_noise_error`, `h2o_vmr_solution_error`, `h2o_vmr_prior` and `h2o_vmr_averaging_kernel`, and the retrieved aerosol |

## Options
`instrument` names the instrument definition (default `SHOW-0-SHS-REP1`), `processing`
the processing version (default `v0.1`) and `acquisition` the acquisition pattern (default
`nominal`); see [Instruments, Processing Versions and Acquisition Patterns](../instruments.md).
The instrument definition sets the orbit altitude and the lines of sight; to change those for one
run, pass `viewing_tangent_altitudes` or `observer_altitude` in `overrides`.  A run is adjusted
further with:

| Input | Passed to |
|---|---|
| `cal_db_cfg` | Overrides of the calibration database settings derived from the instrument definition: arguments of `ideal_calibration_database` such as the detector (`detector_full_well`, `detector_adc_bits`, `detector_read_noise`, `dark_offset`, `dark_offset_spread`, `dark_current`), `fringe_visibility` and a grating tilt (`grating_tilt_rad`, `tilted_grating`, `row_pitch_cm`, `magnification` or `row_edge_height_cm`), plus `detector_gain` and `filter_peak_transmission` |
| `acquisition_cfg` | Overrides of the acquisition pattern, e.g. `{"timing": {"integration_time_s": 4.0}}` |
| `l1a_cfg` | `simulate_l1a`: `noise` (default `True`) and `seed`, and `exposure_time` and `num_averaged_exposures` to replace the acquisition pattern's exposure |
| `l1b_cfg` | Overrides of the processing version's `process_l1a_to_l1b` arguments: `apodization`, `noise_fraction`, and `dark`, the dark estimate subtracted from each image (by default the calibration database's dark offset plus dark current) |
| `l2_cfg` | Overrides of the processing version's retrieval settings, which are themselves overrides of showlib's retrieval configuration (`showlib/l2/retrieval_config.yaml`), e.g. `{"engine": {"num_threads": 4}}`. By default the retrieval uses every CPU core |
| `line_shape_cache` | Directory the L2 line shapes are saved in and loaded from, keyed by the calibration database, apodization and model resolution (default `~/.cache/hawcsimulator/show_line_shapes`, about 2 GB per 200-row instrument); `None` builds them for each run |
| `sk2_kwargs` | `sasktran2.Config` attributes for the front end radiance |

For example `{"l1a_cfg": {"noise": False}}` gives a noise-free simulation.  A different calibration
database can be used with `run(..., overrides={"calibration_database": cal_db})`; it needs one row
per viewing tangent altitude.
