from __future__ import annotations

import sasktran2 as sk
from showlib.l2.optical import h2o_optical_property

from hawcsimulator.steps import Step


class DefaultAtmosphere(Step):
    def _run(self, data: dict, cfg: dict) -> dict:  # noqa: ARG002

        data["atmosphere"] = {
            "rayleigh": sk.constituent.Rayleigh(),
            "h2o": sk.climatology.mipas.constituent(
                "H2O", h2o_optical_property(), climatology="std"
            ),
        }

        return data
