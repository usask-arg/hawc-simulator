(_changelog)=
# Changelog

## Unreleased
- Requires `skretrieval`, `showlib`, and `aliprocessing` 2026.10.0 and `sasktran2` 2026.10.1.  The previously
  released `showlib` was missing `showlib.cal_db`, so the SHOW simulators could not run from a PyPI install
- Development environment is now managed with `uv` only; `pixi` support has been removed
- Release builds use `uv build`
- `xarray`, `pandas`, `scipy`, `astropy`, `pyyaml`, and `packaging` are now declared as dependencies
- Fixed loading the `omps_calipso_era5` atmosphere with recent versions of `xarray`
- Fixed the quickstart examples: the orbit example now uses a daylit measurement inside the atmosphere curtain
  and specifies the observer location, and the examples use the public `sasktran2` constituent attributes
