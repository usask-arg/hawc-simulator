from __future__ import annotations

import sasktran2 as sk

from hawcsimulator.steps import Step


class DefaultAtmosphere(Step):
    def _run(self, data: dict, cfg: dict) -> dict:

        data["atmosphere"] = {
            "rayleigh": sk.constituent.Rayleigh(),
            "o3": sk.climatology.mipas.constituent(
                "O3", sk.optical.O3DBM(), climatology="std"
            ),
            "solar_irradiance": sk.constituent.SolarIrradiance(
                mode="average", resolution=1
            ),
            "albedo": sk.constituent.LambertianSurface(0.3),
        }

        for k, v in cfg.get("constituents", {}).items():
            data["atmosphere"][k] = v

        return data
