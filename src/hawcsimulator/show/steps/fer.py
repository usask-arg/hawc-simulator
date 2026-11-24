from __future__ import annotations

import os

import numpy as np
import sasktran2 as sk
from hamilton.function_modifiers import config, extract_fields
from showlib.cal_db import CalibrationDatabase
from skretrieval.core.sasktranformat import SASKTRANRadiance

from hawcsimulator.datastructures.atmosphere import Atmosphere
from hawcsimulator.datastructures.viewinggeo import ObservationContainer
from hawcsimulator.fer import FERGeneratorBasic


@extract_fields(
    {
        "fer_gen": FERGeneratorBasic,
        "sk2_atmosphere": sk.Atmosphere,
    }
)
def sk2_atm_and_front_end_radiance_gen(
    observation: ObservationContainer,
    atmosphere: Atmosphere,
    altitude_grid: np.ndarray,
    calibration_database: CalibrationDatabase,
    sk2_kwargs: dict | None = None,
) -> dict:
    """
    The SASKTRAN2 atmosphere and radiance generator for the SHOW front end radiance.

    The radiance is calculated on the calibration database's model wavenumber grid,
    which covers the instrument's spectral transmission. MIPAS CH4 is added unless
    the atmosphere already has "ch4", since the L2 retrieval models it.

    Parameters
    ----------
    sk2_kwargs : dict | None, optional
        sasktran2.Config attributes to set, by default None
    """
    if sk2_kwargs is None:
        sk2_kwargs = {}

    fer_gen = FERGeneratorBasic(observation.observation, altitude_grid)

    fer_gen.sk_config.los_refraction = True
    fer_gen.sk_config.multiple_scatter_source = sk.MultipleScatterSource.TwoStream
    fer_gen.sk_config.num_streams = 2
    fer_gen.sk_config.wavelength_batch_size = 128
    fer_gen.sk_config.num_threads = os.cpu_count() or 1

    for k, v in sk2_kwargs.items():
        setattr(fer_gen.sk_config, k, v)

    sk2_atmosphere = sk.Atmosphere(
        model_geometry=fer_gen.model_geo,
        config=fer_gen.sk_config,
        wavenumber_cminv=calibration_database.model_wavenumber(),
        calculate_derivatives=False,
    )

    sk.climatology.us76.add_us76_standard_atmosphere(sk2_atmosphere)

    constituents = dict(atmosphere.constituents)
    if "ch4" not in constituents:
        constituents["ch4"] = sk.climatology.mipas.constituent(
            "CH4", sk.optical.HITRANAbsorber("CH4")
        )
    for k, v in constituents.items():
        sk2_atmosphere[k] = v

    return {
        "fer_gen": fer_gen,
        "sk2_atmosphere": sk2_atmosphere,
    }


@config.when(FER_provided=False)
def front_end_radiance(
    fer_gen: FERGeneratorBasic, sk2_atmosphere: sk.Atmosphere
) -> SASKTRANRadiance:
    return fer_gen.run(sk2_atmosphere)
