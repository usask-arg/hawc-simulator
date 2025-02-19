from __future__ import annotations

import xarray as xr
from showlib.l1b.data import L1bDataSet
from showlib.processing.l1b_to_l2 import process_l1b_to_l2


def l2(
    l1b: L1bDataSet,
    program_of_record: xr.Dataset,
    calibration_database: xr.Dataset,
    l2_cfg: dict | None = None,
) -> list:
    if l2_cfg is None:
        l2_cfg = {}

    return process_l1b_to_l2(l1b, program_of_record, calibration_database, **l2_cfg)
