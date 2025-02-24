from __future__ import annotations

import xarray as xr
from aliprocessing.l1b.data import L1bImage
from aliprocessing.processing.l1b_to_l2 import process_l1b_to_l2_image


def l2(
    l1b: L1bImage,
    program_of_record: xr.Dataset,
    calibration_database: xr.Dataset,
    l2_cfg: dict | None = None,
) -> xr.Dataset:
    if l2_cfg is None:
        l2_cfg = {}

    return process_l1b_to_l2_image(
        l1b, program_of_record.isel(time=0), calibration_database, **l2_cfg
    )
