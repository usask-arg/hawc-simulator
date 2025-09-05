from __future__ import annotations

import numpy as np
import sasktran2 as sk
from hamilton.function_modifiers import extract_fields
from skretrieval.core.sasktranformat import SASKTRANRadiance

from hawcsimulator.datastructures.atmosphere import Atmosphere
from hawcsimulator.datastructures.viewinggeo import ObservationContainer
from hawcsimulator.fer import FERGeneratorBasic
from hamilton.function_modifiers import config


@extract_fields(
    {
        # "front_end_radiance": SASKTRANRadiance,
        "fer_gen": FERGeneratorBasic,
        "sk2_atmosphere": sk.Atmosphere,
    }
)
def sk2_atm_and_front_end_radiance_gen(
    observation: ObservationContainer,
    atmosphere: Atmosphere,
    altitude_grid: np.ndarray,
    sk2_kwargs: dict | None = None,
) -> dict:
    if sk2_kwargs is None:
        sk2_kwargs = {}

    # Construct the FER generator
    fer_gen = FERGeneratorBasic(observation.observation, altitude_grid)

    # Engine properties
    fer_gen.sk_config.num_stokes = 3
    fer_gen.sk_config.stokes_basis = sk.StokesBasis.Observer
    fer_gen.sk_config.multiple_scatter_source = (
        sk.MultipleScatterSource.DiscreteOrdinates
    )
    fer_gen.sk_config.num_streams = 8
    fer_gen.sk_config.input_validation_mode = sk.InputValidationMode.Disabled

    for k, v in sk2_kwargs.items():
        setattr(fer_gen.sk_config, k, v)

    sk2_atmosphere = sk.Atmosphere(
        model_geometry=fer_gen.model_geo,
        config=fer_gen.sk_config,
        wavelengths_nm=observation.observation.sample_wavelengths()["measurement"],
        calculate_derivatives=False,
    )

    sk.climatology.us76.add_us76_standard_atmosphere(sk2_atmosphere)

    for k, v in atmosphere.constituents.items():
        sk2_atmosphere[k] = v

    # # Run the FER generator
    # rad = fer_gen.run(sk2_atmosphere)

    return {
        "fer_gen": fer_gen,
        "sk2_atmosphere": sk2_atmosphere,
    }

@config.when(FER_provided=False)
def front_end_radiance(fer_gen: FERGeneratorBasic, sk2_atmosphere: sk.Atmosphere)->SASKTRANRadiance:
    return fer_gen.run(sk2_atmosphere)