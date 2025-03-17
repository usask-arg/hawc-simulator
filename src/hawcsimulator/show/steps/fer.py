from __future__ import annotations

import numpy as np
import sasktran2 as sk
from hamilton.function_modifiers import extract_fields
from skretrieval.core.sasktranformat import SASKTRANRadiance

from hawcsimulator.datastructures.atmosphere import Atmosphere
from hawcsimulator.datastructures.viewinggeo import ObservationContainer
from hawcsimulator.fer import FERGeneratorBasic


@extract_fields(
    {
        "front_end_radiance": SASKTRANRadiance,
        "sk2_atmosphere": sk.Atmosphere,
    }
)
def sk2_atm_and_front_end_radiance(
    observation: ObservationContainer,
    atmosphere: Atmosphere,
    altitude_grid: np.ndarray,
    sk2_kwargs: dict | None = None,
) -> dict:
    if sk2_kwargs is None:
        sk2_kwargs = {}

    # Construct the FER generator
    fer_gen = FERGeneratorBasic(observation.observation, altitude_grid)

    fer_gen.sk_config.los_refraction = True
    fer_gen.sk_config.multiple_scatter_source = (
        sk.MultipleScatterSource.DiscreteOrdinates
    )
    fer_gen.sk_config.num_streams = 2

    for k, v in sk2_kwargs:
        setattr(fer_gen.sk_config, k, v)

    sk2_atmosphere = sk.Atmosphere(
        model_geometry=fer_gen.model_geo,
        config=fer_gen.sk_config,
        wavenumber_cminv=np.arange(7295, 7340, 0.01),
        calculate_derivatives=False,
    )

    sk.climatology.us76.add_us76_standard_atmosphere(sk2_atmosphere)

    for k, v in atmosphere.constituents.items():
        sk2_atmosphere[k] = v

    # Run the FER generator
    rad = fer_gen.run(sk2_atmosphere)

    return {
        "front_end_radiance": rad,
        "sk2_atmosphere": sk2_atmosphere,
    }
