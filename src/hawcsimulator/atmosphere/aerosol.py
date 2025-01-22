from __future__ import annotations

import numpy as np
import sasktran2 as sk


def background_aerosol_extinction(alts: np.array):
    return sk.test_util.scenarios.test_aerosol_constituent(alts)._extinction_per_m
