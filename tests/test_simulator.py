from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hawcsimulator.simulator import Simulator

INPUT = {
    "viewing_tangent_altitudes": np.arange(10000.0, 40001.0, 1000.0),
    "observer_altitude": 450000.0,
    "sample_wavelengths": np.array([745.0]),
    "tangent_latitude": 30.0,
    "tangent_longitude": 0.0,
    "tangent_solar_zenith_angle": 60.0,
    "tangent_solar_azimuth_angle": 0.0,
    "time": pd.Timestamp("2022-01-01T12:00:00Z"),
}


def test_run():
    data = Simulator().run(["observation", "atmosphere"], INPUT)

    assert set(data) == {"observation", "atmosphere"}


def test_input_that_names_a_computed_node_is_rejected():
    # Hamilton would silently use the computed atmosphere instead
    with pytest.raises(ValueError, match="pass them in `overrides`"):
        Simulator().run(["observation"], {**INPUT, "atmosphere": None})


def test_unused_input_is_rejected():
    # e.g. an input of the old ALI simulators, which Hamilton would silently ignore
    with pytest.raises(ValueError, match=r"\['polarization_states'\] are not used"):
        Simulator().run(["observation"], {**INPUT, "polarization_states": ["I"]})
