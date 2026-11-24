from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

# Runs a SHOW retrieval and then an ALI retrieval in one process, as a notebook that
# uses both simulators would
SCRIPT = textwrap.dedent(
    """
    import numpy as np
    import pandas as pd
    import sasktran2 as sk
    from aliprocessing.l2.optical import aerosol_median_radius_db

    from hawcsimulator.ali import ALISimulator
    from hawcsimulator.show import SHOWSimulator

    geometry = {
        "tangent_latitude": 30.0,
        "tangent_longitude": 0.0,
        "tangent_solar_zenith_angle": 60.0,
        "tangent_solar_azimuth_angle": 120.0,
        "time": pd.Timestamp("2022-01-01T12:00:00Z"),
    }

    show = SHOWSimulator().run(
        ["l2"],
        {
            **geometry,
            "l1a_cfg": {"noise": False},
            "l2_cfg": {"engine": {"num_threads": 4}},
        },
        overrides={"viewing_tangent_altitudes": np.arange(5000.0, 35001.0, 1000.0)},
    )
    assert np.all(np.isfinite(show["l2"].ds["h2o_vmr"]))

    altitude = np.arange(0.0, 65001.0, 1000.0)
    extinction = 1e-8 + 2e-6 * np.exp(-0.5 * ((altitude - 20e3) / 4e3) ** 2)
    aerosol = sk.constituent.ExtinctionScatterer(
        aerosol_median_radius_db(),
        altitude,
        extinction,
        745.0,
        "extend",
        median_radius=np.full(len(altitude), 80.0),
    )
    ali = ALISimulator().run(
        ["l2"],
        {
            **geometry,
            "constituents": {"aerosol": aerosol},
            "l1a_cfg": {"noise": False},
            "l2_cfg": {"minimizer": {"verbose": 0}},
        },
    )
    retrieved = ali["l2"]["aerosol_extinction"].isel(observation_set=0, across_track=0)
    np.testing.assert_allclose(
        retrieved.interp(altitude=[15e3, 20e3, 25e3]),
        np.interp([15e3, 20e3, 25e3], altitude, extinction),
        rtol=0.1,
    )
    """
)


@pytest.mark.slow
def test_ali_after_show_in_one_process():
    """
    showlib and aliprocessing both register "stratospheric_aerosol" with skretrieval,
    showlib on its SHOWRetrieval subclass. With a skretrieval that shares one registry
    across Retrieval subclasses, the ALI retrieval after a SHOW import used SHOW's
    1300-1400 nm aerosol database and sasktran2 aborted the interpreter, so the
    script runs in a child process.
    """
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        capture_output=True,
        text=True,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
        timeout=3600,
        check=False,
    )

    assert result.returncode == 0, result.stderr[-5000:]
