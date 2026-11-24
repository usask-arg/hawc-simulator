from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sasktran2 as sk
from showlib.l2.retrieval import stratospheric_aerosol_optical_property

FORCED = {
    "tangent_latitude": 30.0,
    "tangent_longitude": 0.0,
    "tangent_solar_zenith_angle": 60.0,
    "tangent_solar_azimuth_angle": 0.0,
    "time": pd.Timestamp("2022-01-01T12:00:00Z"),
}
ALTITUDE = np.arange(0.0, 65001.0, 1000.0)
EXTINCTION = 1e-8 + 1e-6 * np.exp(-0.5 * ((ALTITUDE - 18e3) / 3e3) ** 2)


def _aerosol(wavelength_nm):
    return sk.constituent.ExtinctionScatterer(
        stratospheric_aerosol_optical_property(),
        ALTITUDE,
        EXTINCTION,
        wavelength_nm,
        "extend",
        median_radius=np.full(len(ALTITUDE), 80.0),
    )


def test_show_program_of_record_holds_the_true_aerosol():
    from hawcsimulator.show import SHOWSimulator

    por = SHOWSimulator().run(
        ["program_of_record"],
        {**FORCED, "constituents": {"aerosol": _aerosol(1350.0)}},
    )["program_of_record"]

    altitude = por["altitude"].to_numpy()
    np.testing.assert_allclose(
        por["aerosol_extinction"].isel(time=0),
        np.interp(altitude, ALTITUDE, EXTINCTION, right=0.0),
    )

    # The default atmosphere has no aerosol, so the retrieval uses the pre-fit's
    por = SHOWSimulator().run(["program_of_record"], FORCED)["program_of_record"]
    assert "aerosol_extinction" not in por


def test_program_of_record_aerosol_must_be_at_the_requested_wavelength():
    from hawcsimulator.show import SHOWSimulator

    with pytest.raises(ValueError, match="745.0 nm, not the 1350.0 nm"):
        SHOWSimulator().run(
            ["program_of_record"],
            {**FORCED, "constituents": {"aerosol": _aerosol(745.0)}},
        )


def test_ali_program_of_record_has_no_aerosol():
    from hawcsimulator.ali import ALISimulator

    por = ALISimulator().run(["program_of_record"], FORCED)["program_of_record"]
    assert "aerosol_extinction" not in por
