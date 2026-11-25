from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from showlib.cal_db.ideal import ideal_calibration_database

from hawcsimulator.geometry.orbit import NOMINAL_ORBIT, time_at_tangent_latitude
from hawcsimulator.show import SHOWSimulator
from hawcsimulator.show.steps.instrument import _radiance_by_row

FORCED = {
    "tangent_latitude": 30.0,
    "tangent_longitude": 0.0,
    "tangent_solar_zenith_angle": 60.0,
    "tangent_solar_azimuth_angle": 20.0,
    "time": pd.Timestamp("2022-01-01T12:00:00Z"),
}

TEST_DEFINITIONS = Path(__file__).parent / "data"

# Fewer lines of sight than the default, to keep the fast tests fast
TANGENT_ALTITUDES = np.arange(10000.0, 30001.0, 1000.0)


def test_calibration_database_has_a_row_per_line_of_sight():
    data = SHOWSimulator().run(["calibration_database"])
    assert data["calibration_database"].num_rows == 200

    data = SHOWSimulator().run(
        ["calibration_database"],
        overrides={"viewing_tangent_altitudes": TANGENT_ALTITUDES},
    )
    assert data["calibration_database"].num_rows == len(TANGENT_ALTITUDES)


def test_cal_db_cfg_sets_the_instrument():
    # The dark signal, fringe visibility and grating tilt are showlib's
    # ideal_calibration_database options
    data = SHOWSimulator().run(
        ["calibration_database", "l1b"],
        {
            **FORCED,
            "cal_db_cfg": {
                "dark_current": 20.0,
                "fringe_visibility": 0.8,
                "grating_tilt_rad": 1e-4,
                "row_pitch_cm": 0.01,
            },
            "l1a_cfg": {"noise": False},
        },
        overrides={"viewing_tangent_altitudes": TANGENT_ALTITUDES},
    )

    cal_db = data["calibration_database"].ds
    np.testing.assert_allclose(cal_db["dark_current"], 20.0)
    np.testing.assert_allclose(cal_db["fringe_visibility"], 0.8)
    assert np.ptp(cal_db["fringe_phase"].to_numpy()) > 0
    assert np.all(np.isfinite(data["l1b"].ds["radiance"]))


def test_l1a_and_l1b():
    data = SHOWSimulator().run(
        ["l1a", "l1b"],
        {**FORCED, "l1a_cfg": {"seed": 1}},
        overrides={"viewing_tangent_altitudes": TANGENT_ALTITUDES},
    )

    l1a = data["l1a"].ds.isel(time=0)
    # SHOW-0-SHS-REP1 has 1024 interferogram samples
    assert l1a["image"].shape == (len(TANGENT_ALTITUDES), 1024)
    np.testing.assert_allclose(
        l1a["tangent_altitude"], TANGENT_ALTITUDES, rtol=0, atol=1.0
    )
    np.testing.assert_allclose(l1a["solar_zenith_angle"], 60.0, atol=1e-6)
    np.testing.assert_allclose(l1a["relative_solar_azimuth_angle"], 20.0, atol=1e-6)
    # The automatic exposure is split into exposures no longer than the detector's
    # 92.5 ms maximum, which the image averages
    assert float(l1a["exposure_time"]) <= 0.0925
    assert int(l1a["num_averaged_exposures"]) > 1
    # Noisy images are averages of digitized exposures
    total = l1a["image"] * l1a["num_averaged_exposures"]
    np.testing.assert_allclose(total, np.rint(total), rtol=0, atol=1e-6)
    assert bool(data["l1a"].simulation["noise_enabled"])

    l1b = data["l1b"].ds.isel(time=0)
    assert np.all(np.isfinite(l1b["radiance"]))
    assert np.all(l1b["radiance_noise"] > 0)

    # Both record what produced them
    for product in (data["l1a"].ds, data["l1b"].ds):
        assert product.attrs["hawcsimulator_instrument"] == "SHOW-0-SHS-REP1"
        assert product.attrs["hawcsimulator_processing"] == "v0.1"
        assert product.attrs["hawcsimulator_acquisition"] == "nominal"
        assert "showlib_version" in product.attrs

    # The nominal acquisition integrates 2 s, averaged from exposures of at most
    # the detector's 92.5 ms
    exposure = float(l1a["exposure_time"])
    num_exposures = int(l1a["num_averaged_exposures"])
    assert exposure <= 0.0925
    assert exposure * num_exposures == pytest.approx(2.0)


def test_representative_definition():
    data = SHOWSimulator().run(["calibration_database"])
    cal_db = data["calibration_database"]

    np.testing.assert_allclose(cal_db.ds["fringe_visibility"], 0.8)
    # 0.13 cm-1 resolution: 1 / (2 x maximum OPD)
    opd = cal_db.ds["opd"].isel(los_edge=0).to_numpy()
    assert 1 / (2 * np.abs(opd).max()) == pytest.approx(0.13, rel=0.01)


def test_interferometer_definition():
    # The design form of a SHOW definition (synthetic values), with a grating tilt
    data = SHOWSimulator().run(
        ["calibration_database", "instrument_definition"],
        {"instrument": str(TEST_DEFINITIONS / "SHOW-0-SHS-TESTDESIGN1.yaml")},
    )
    cal_db = data["calibration_database"].ds
    interferometer = data["instrument_definition"]["interferometer"]

    # One column per detector pixel across the illuminated grating width
    assert cal_db.sizes["pixelcolumn"] == 1000
    np.testing.assert_allclose(cal_db["fringe_visibility"], 0.9)
    # fringe_phase is the tilt's phase at the Littrow wavenumber across the rows:
    # 2 pi sigma 2 eps H
    expected = (
        interferometer["littrow_wavenumber_cminv"]
        * 2
        * interferometer["grating_tilt_rad"]
        * interferometer["illuminated_grating_height_cm"]
    )
    fringes = np.ptp(cal_db["fringe_phase"].to_numpy()) / (2 * np.pi)
    assert fringes == pytest.approx(expected, rel=0.01)


def test_front_end_radiance_must_be_on_the_model_grid():
    cal_db = ideal_calibration_database(3)
    wavenumber = cal_db.model_wavenumber()
    fer = SimpleNamespace(
        data=xr.Dataset(
            {"radiance": (["spectral_grid", "los"], np.ones((len(wavenumber), 3)))},
            coords={"wavenumber_cminv": ("spectral_grid", wavenumber + 0.005)},
        )
    )

    with pytest.raises(ValueError, match="model wavenumber grid"):
        _radiance_by_row(fer, cal_db)


def _assert_retrieves_the_truth(data):
    l2 = data["l2"].ds.isel(time=0)
    h2o = data["atmosphere"].constituents["h2o"]
    stratosphere = l2.sel(altitude=slice(17000, 30000))
    truth = np.interp(
        stratosphere["altitude"].to_numpy(),
        np.asarray(h2o.altitudes_m),
        np.asarray(h2o.vmr),
    )

    np.testing.assert_allclose(stratosphere["h2o_vmr"], truth, rtol=0.05)
    assert np.all(stratosphere["h2o_vmr_noise_error"] > 0)
    assert l2["h2o_vmr_averaging_kernel"].shape == (len(l2["altitude"]),) * 2
    assert (
        data["l2"].ds.attrs["hawcsimulator_instrument"]
        == SHOWSimulator.default_instrument
    )


@pytest.mark.slow
def test_noise_free_retrieval_returns_the_truth():
    data = SHOWSimulator().run(
        ["l2", "atmosphere"],
        {
            **FORCED,
            "l1a_cfg": {"noise": False},
        },
    )

    _assert_retrieves_the_truth(data)


@pytest.mark.slow
def test_noise_free_retrieval_on_the_orbit_returns_the_truth():
    time = time_at_tangent_latitude(NOMINAL_ORBIT, 45.0, np.datetime64("2026-06-21"))
    data = SHOWSimulator().run(
        ["l2", "atmosphere"],
        {"time": time, "l1a_cfg": {"noise": False}},
        config={"observation_method": "orbit"},
    )

    _assert_retrieves_the_truth(data)


def test_line_shapes_are_cached(tmp_path):
    from hawcsimulator.show.steps.l2 import cached_line_shapes

    cal_db = ideal_calibration_database(3)
    built = cached_line_shapes(cal_db, 10.0, 0.05, tmp_path)
    (entry,) = list(tmp_path.iterdir())

    loaded = cached_line_shapes(cal_db, 10.0, 0.05, tmp_path)
    assert list(tmp_path.iterdir()) == [entry]
    np.testing.assert_array_equal(loaded.matrix(1), built.matrix(1))

    # Different settings are a different entry
    cached_line_shapes(cal_db, None, 0.05, tmp_path)
    assert len(list(tmp_path.iterdir())) == 2
