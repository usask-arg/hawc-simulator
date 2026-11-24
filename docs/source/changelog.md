(_changelog)=
# Changelog

## Unreleased
- **Breaking:** the SHOW and ALI simulators are replaced by `hawcsimulator.show.SHOWSimulator` and
  `hawcsimulator.ali.ALISimulator`, built on the instrument models and processing in `showlib` and
  `aliprocessing`: radiance is simulated to L1A detector data, processed to L1B, and retrieved.
  `IdealSHOWSimulator`, the `showinstrument`-based `SHOWSimulator`, both `IdealALISimulator`s and
  `ALIPhase0Simulator` are removed, along with `hawcsimulator.noise`.  A noise-free simulation is
  `l1a_cfg={"noise": False}`
- Instruments are described by named instrument definitions (`SHOW-0-SHS-REP1`, `ALI-0-FWHEEL-REP1`),
  the processing by versions (`v0.1`) and the pointing and exposure by acquisition patterns (`nominal`),
  chosen with the `instrument`, `processing` and `acquisition` inputs; the outputs record them, and the
  library versions, in their attributes.  SHOW processing v0.1 uses lines of sight up to 35 km in the
  retrieval (showlib's default is 30 km) and holds the surface albedo at its prior.  See
  [Instruments, Processing Versions and Acquisition Patterns](instruments.md)
- **Breaking:** the defaults are representative phase 0 instruments on a Raptor Owl 1280 detector, with
  the phase 0 designs' coverage, sampling and bands and close to their performance: SHOW has 200 lines
  of sight from 0 to 45 km, 0.13 cm⁻¹ resolution and fringe visibility 0.8, and integrates 2 s per image;
  ALI views from 450 km with three polarization sub-images, 5 nm filters from the filter specification
  and Thorlabs LPVIS polarizers.  Both average exposures of at most the detector's 92.5 ms, with
  configurable read noise and dead time.  `viewing_tangent_altitudes` and `observer_altitude` (SHOW) and
  `observer_altitude` (ALI) come from the instrument definition and are changed with `overrides`; ALI's
  `ali_instrument` input is replaced by `cal_db_cfg`, overrides of the `SimpleInstrument` fields
- **Breaking:** ALI's pointing and timing come from the acquisition pattern (`acquisition`,
  `acquisition_cfg`): the boresight tangent altitude and roll, the number of sets, the exposure and the filter-wheel change time.  `l1a_cfg` keeps `noise`, `seed`,
  `exposure_time` and `num_averaged_exposures`; `num_sets`, `fill` and `frame_interval` move to the
  acquisition pattern, and the L1A frame times follow from the exposures, the dead time and the filter
  changes
- **Breaking:** the limb observation is built from ECEF lines of sight
  (`hawcsimulator.geometry.ecef.ECEFLimbGeometry`) instead of sasktran2's `LimbVertical`, which fixes the
  viewing azimuth with time-based solar angles (it was passed in radians where degrees were expected).
  The lines of sight look along the new `los_azimuth` input (default north), or towards the tangent point
  from `observer_latitude` and `observer_longitude`, which no longer need to be given for time-based solar
  angles.  Forced solar angles apply at every tangent point
- **Breaking:** `l2` is in the libraries' L2 formats, checked against their schemas: for SHOW a
  `showlib.l2.data.L2DataSet` along `time` (`h2o_vmr_noise_error`, `h2o_vmr_solution_error`,
  `h2o_vmr_averaging_kernel`), and for ALI a granule along `observation_set` and `across_track`
  (`aerosol_extinction`, `aerosol_median_radius`, `lower_altitude_limit`, ...).  `l1b_image_cfg` can
  retrieve several observation sets and across-track bins, or `"all"`
- **Breaking:** SHOW's `l2_cfg` overrides showlib's retrieval configuration (e.g.
  `{"engine": {"num_threads": 4}}`)
- **Breaking:** ALI no longer takes `sample_wavelengths` or `polarization_states`; the channels come from
  the instrument, and L2 uses radiance and degree of linear polarization
- **Breaking:** ALI's `l2_cfg` overrides aliprocessing's retrieval configuration (e.g.
  `{"minimizer": {"verbose": 0}}`), and the default channels are aliprocessing's: 525, 745, 869, 1020 and 1560 nm
- The SHOW simulation includes showlib's dark signal (offset and dark current, subtracted again in L1A to L1B),
  and `cal_db_cfg` can set the fringe visibility and a grating tilt
- The ALI front end radiance uses `aliprocessing.solar.ChannelSolarIrradiance`, the solar spectrum averaged
  through each channel's spectral response, as its solar irradiance, the same one the ALI L2 retrieval uses
- New `truth` output, the true atmospheric state as an `xarray.Dataset`, and `l2_truth`, the truth on the
  L2's grid with the L2's names, so the retrieval error is `l2 - l2_truth`
- Other installed packages can add instrument, processing and acquisition definitions through the
  `hawcsimulator.definitions` entry point group, for definitions that are not public; outputs record
  each definition's source
- The L2 retrievals limit BLAS to one thread (`threadpoolctl`), which competed with SASKTRAN2's threads;
  `OPENBLAS_NUM_THREADS` no longer needs to be set
- `Simulator.run` raises an error for inputs it would otherwise ignore: an input that names a computed
  output (use `overrides` instead), or an input nothing uses
- `import hawcsimulator` no longer imports the SHOW and ALI simulators; import them from
  `hawcsimulator.show` and `hawcsimulator.ali`
- The program of record's `tropopause_altitude` is now computed; it was always NaN
- SHOW's L2 line shapes are saved once per calibration database and loaded on later runs (`line_shape_cache`,
  by default under `~/.cache/hawcsimulator`), which takes a 200-row retrieval from about 190 s to 70 s
- The program of record can hold the simulation atmosphere's aerosol extinction (`por_aerosol_wavelength_nm`); the
  SHOW simulator does so by default for now, so its retrieval starts from the true aerosol when there is one
- The `default` atmosphere adds the MIPAS H2O profile when an `h2o_optical_property` is given
- Requires `showlib`, `aliprocessing` and `skretrieval` 2026.11.0 and `sasktran2` 2026.10.1
- Development environment is now managed with `uv` only; `pixi` support has been removed
- Release builds use `uv build`
- `xarray`, `pandas`, `scipy`, `astropy`, `pyyaml`, and `packaging` are now declared as dependencies
- Fixed loading the `omps_calipso_era5` atmosphere with recent versions of `xarray`
- Fixed the quickstart examples: the orbit example now uses a daylit measurement inside the atmosphere curtain
  and specifies the observer location, and the examples use the public `sasktran2` constituent attributes
