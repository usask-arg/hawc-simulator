from __future__ import annotations

import numpy as np
from showlib.l2.optical import h2o_optical_property
from showlib.l2.retrieval import stratospheric_aerosol_optical_property

import hawcsimulator.show.steps.fer as fer
import hawcsimulator.show.steps.instrument as instrument
import hawcsimulator.show.steps.l2 as l2
import hawcsimulator.steps.program_of_record as program_of_record
import hawcsimulator.steps.truth as truth
from hawcsimulator.show.steps.l2 import default_line_shape_cache
from hawcsimulator.simulator import Simulator


class SHOWSimulator(Simulator):
    """
    The SHOW simulator: front end radiance, showlib's instrument model (L1A
    interferograms and L1B spectra), and showlib's L2 H2O retrieval.

    The ``instrument`` input names the instrument definition (default
    SHOW-0-SHS-REP1), ``processing`` the processing version (default v0.1) and
    ``acquisition`` the acquisition pattern (default nominal: 2 s images); see
    hawcsimulator.definitions. Outputs include ``atmosphere``, ``observation``,
    ``calibration_database``, ``front_end_radiance``, ``sk2_atmosphere``, ``l1a``,
    ``l1b``, ``program_of_record``, ``l2``, ``truth`` (the true atmospheric state), ``l2_truth`` (the truth on the L2's grid) and ``provenance``. The ``cal_db_cfg``,
    ``acquisition_cfg``, ``l1a_cfg``, ``l1b_cfg`` and ``l2_cfg`` inputs adjust a
    run; for example
    ``l1a_cfg={"noise": False}`` gives a noise-free simulation.
    """

    # The instrument definition, processing version and acquisition pattern a run
    # uses unless the input sets "instrument", "processing" or "acquisition"; a
    # subclass can change them
    default_instrument = "SHOW-0-SHS-REP1"
    default_processing = "v0.1"
    default_acquisition = "nominal"

    def __init__(self) -> None:
        super().__init__()

        self._modules.append(fer)
        self._modules.append(instrument)
        self._modules.append(program_of_record)
        self._modules.append(truth)
        self._modules.append(l2)

    def _initialize_data(self) -> dict:
        return {
            "instrument": self.default_instrument,
            "processing": self.default_processing,
            "acquisition": self.default_acquisition,
            "altitude_grid": np.arange(0, 65001, 1000.0),
            "h2o_optical_property": h2o_optical_property(),
            "aerosol_optical_property": stratospheric_aerosol_optical_property(),
            "aerosol_kwargs": {"extinction_wavelength_nm": 1350.0},
            # For now the retrieval starts from the true aerosol, when the atmosphere
            # has one, at the wavelength of showlib's aerosol state: showlib's pre-fit
            # aerosol fails when the atmosphere's H2O differs from its climatology
            "por_aerosol_wavelength_nm": 1350.0,
            # L2 line shapes are built once per calibration database and reused
            "line_shape_cache": default_line_shape_cache(),
        }
