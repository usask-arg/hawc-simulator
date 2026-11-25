from __future__ import annotations

import numpy as np
from aliprocessing.l2.optical import aerosol_median_radius_db

import hawcsimulator.ali.steps.fer as fer
import hawcsimulator.ali.steps.instrument as instrument
import hawcsimulator.ali.steps.l2 as l2
import hawcsimulator.steps.program_of_record as program_of_record
import hawcsimulator.steps.sampling as sampling
import hawcsimulator.steps.truth as truth
from hawcsimulator.simulator import Simulator


class ALISimulator(Simulator):
    """
    The ALI simulator: polarized front end radiance, aliprocessing's simple
    instrument model (L1A detector frames and L1B radiance and polarization), and
    aliprocessing's L2 aerosol retrieval.

    The ``instrument`` input names the instrument definition (default
    ALI-0-FWHEEL-REP1), ``processing`` the processing version (default v0.1) and
    ``acquisition`` the acquisition pattern (default nominal: a limb stare);
    see hawcsimulator.definitions. Outputs include ``atmosphere``, ``observation``,
    ``ali_instrument``, ``calibration_database``, ``front_end_radiance``,
    ``sk2_atmosphere``, ``pointing``, ``scene``, ``l1a``, ``l1b``,
    ``program_of_record``, ``l2``, ``truth`` (the true atmospheric state), ``l2_truth`` (the truth on the L2's grid) and ``provenance``. The ``cal_db_cfg`` (overrides
    of aliprocessing.cal_db.SimpleInstrument fields), ``l1a_cfg``, ``l1b_cfg``,
    ``l1b_image_cfg`` and ``l2_cfg`` inputs adjust a run; for example
    ``l1a_cfg={"noise": False}`` gives a noise-free simulation.
    """

    # The instrument definition, processing version and acquisition pattern a run
    # uses unless the input sets "instrument", "processing" or "acquisition"; a
    # subclass can change them
    default_instrument = "ALI-0-FWHEEL-REP1"
    default_processing = "v0.1"
    default_acquisition = "nominal"

    def __init__(self) -> None:
        super().__init__()
        self._modules.append(fer)
        self._modules.append(instrument)
        self._modules.append(program_of_record)
        self._modules.append(truth)
        self._modules.append(sampling)
        self._modules.append(l2)

    def _initialize_data(self) -> dict:
        return {
            "instrument": self.default_instrument,
            "processing": self.default_processing,
            "acquisition": self.default_acquisition,
            # The scene has to cover the L1B altitude grid (0-50 km) widened by the
            # resampling kernel
            "viewing_tangent_altitudes": np.arange(-3000.0, 56001.0, 500.0),
            "altitude_grid": np.arange(0, 65001, 1000.0),
            "aerosol_optical_property": aerosol_median_radius_db(),
            "aerosol_kwargs": {"extinction_wavelength_nm": 745.0},
        }
