from __future__ import annotations

import numpy as np
import sasktran2 as sk

from hawcsimulator.fer import FERGeneratorBasic
from hawcsimulator.steps import Step


class GenerateFER(Step):
    def _run(self, data: dict, cfg: dict) -> dict:
        observation = data["observation"]

        # Construct the FER generator
        fer_gen = FERGeneratorBasic(
            observation, cfg.get("altitude_grid", np.arange(0.0, 65001, 1000.0))
        )

        # Engine properties
        fer_gen.sk_config.num_stokes = 3
        fer_gen.sk_config.stokes_basis = sk.StokesBasis.Observer
        fer_gen.sk_config.multiple_scatter_source = (
            sk.MultipleScatterSource.DiscreteOrdinates
        )
        fer_gen.sk_config.num_streams = 8
        fer_gen.sk_config.input_validation_mode = sk.InputValidationMode.Disabled

        for k, v in cfg.get("model_kwargs", {}).items():
            setattr(fer_gen.sk_config, k, v)

        atmosphere = sk.Atmosphere(
            model_geometry=fer_gen.model_geo,
            config=fer_gen.sk_config,
            **cfg.get("spectral_grid", {"wavelengths_nm": data["sample_wavelengths"]}),
            calculate_derivatives=False,
        )

        sk.climatology.us76.add_us76_standard_atmosphere(atmosphere)

        for k, v in data["atmosphere"].items():
            atmosphere[k] = v

        # Run the FER generator
        rad = fer_gen.run(atmosphere)

        data["fer"] = rad
        data["sk_atmosphere"] = atmosphere

        return data

    def _validate_data(self, data: dict):
        assert "observation" in data
        assert "atmosphere" in data
