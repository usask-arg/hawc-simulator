from __future__ import annotations

import numpy as np
import pandas as pd
import sasktran2 as sk
import xarray as xr
from showlib.l2.optical import h2o_optical_property
from showlib.processing.l1b_to_l2 import process_l1b_to_l2
from skretrieval.util import configure_log

from hawcsimulator.fer import FERGeneratorBasic
from hawcsimulator.geometry.observation import SimulatedObservationGeometry
from hawcsimulator.show.calibration import (
    calibration_database,
)
from hawcsimulator.show.inst_model import L1bGeneratorILS
from hawcsimulator.show.por import por_from_atmosphere


def test_chain():
    configure_log()

    # Load in the calibration database
    cal_db = xr.open_dataset(calibration_database("ideal", "v1"))

    # Define our observation to generate the front end radiance
    tan_alts = np.arange(0.0, 32000, 250.0)
    obs_time = pd.Timestamp("2021-01-01T12:00:00Z")
    viewing_geo = sk.viewinggeo.LimbVertical.from_tangent_parameters(
        solar_handler=sk.solar.SolarGeometryHandlerAstropy(),
        tangent_altitudes=tan_alts,
        tangent_latitude=30,
        tangent_longitude=20,
        time=obs_time,
        observer_altitude=200000,
        viewing_azimuth=0.0,
    )

    observation = SimulatedObservationGeometry(
        viewing_geo=viewing_geo,
        sample_wavenumber=cal_db["sample_wavenumber"].to_numpy(),
    )

    # Construct the FER generator
    alt_grid = np.arange(0.0, 65001, 1000.0)
    fer_gen = FERGeneratorBasic(observation, alt_grid)

    # Set up the atmosphere for the calculation
    atmo = sk.Atmosphere(
        fer_gen.model_geo,
        fer_gen.sk_config,
        wavenumber_cminv=np.arange(7295, 7340, 0.02),
        calculate_derivatives=False,
    )

    sk.climatology.us76.add_us76_standard_atmosphere(atmo)

    atmo["rayleigh"] = sk.constituent.Rayleigh()
    atmo["h2o"] = sk.climatology.mipas.constituent(
        "H2O", h2o_optical_property(), climatology="std"
    )

    # Run the FER generator
    rad = fer_gen.run(atmo)

    # Create our instrument model and generate the L1b data
    l1b_gen = L1bGeneratorILS(cal_db, observation)
    l1b = l1b_gen.run(rad)

    # Create the POR data from our simulation atmosphere
    por = por_from_atmosphere(atmo, obs_time)

    # Run the L2 processing
    process_l1b_to_l2(l1b, por, cal_db)
