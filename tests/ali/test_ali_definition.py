from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hawcsimulator.ali.definition import build_calibration_database, simple_instrument
from hawcsimulator.definitions import available, load_instrument

TEST_DEFINITIONS = Path(__file__).parent / "data"


@pytest.mark.parametrize(
    "name",
    [
        *available("hawcsimulator.ali", "instruments"),
        # A pixel geometry (synthetic)
        str(TEST_DEFINITIONS / "ALI-0-FWHEEL-TESTGEOM1.yaml"),
    ],
)
def test_instruments_build_valid_calibration_databases(name):
    definition = load_instrument("hawcsimulator.ali", name)
    name = definition["name"]

    # build_calibration_database checks the database against aliprocessing's schema
    cal_db = build_calibration_database(definition, simple_instrument(definition))

    assert cal_db.sizes["detector_row"] == 1024
    assert cal_db.sizes["detector_column"] == 1280
    assert cal_db.attrs["hawcsimulator_instrument"] == name


def test_pixel_geometry():
    definition = load_instrument(
        "hawcsimulator.ali", str(TEST_DEFINITIONS / "ALI-0-FWHEEL-TESTGEOM1.yaml")
    )
    instrument = simple_instrument(definition)
    cal_db = build_calibration_database(definition, instrument)

    analyzer = cal_db["pixel_analyzer"].to_numpy()
    elevation = cal_db["pixel_elevation_angle"].to_numpy()
    azimuth = cal_db["pixel_azimuth_angle"].to_numpy()
    ifov = 0.004068

    # The sub-images are in row order, with the definition's polarizer angles
    assert instrument.analyzer_angles_deg == (60.0, 0.0, -60.0)
    np.testing.assert_allclose(
        cal_db["q_sensitivity"].isel(spectral_channel=0, detector_column=640)[
            [163, 511, 859]
        ],
        # Channel 0 (525 nm) is outside the LPVIS polarizers' specified range,
        # where the extinction ratio is 100:1
        99 / 101 * np.cos(np.deg2rad([120.0, 0.0, -120.0])),
    )
    for k, (first, last) in enumerate([(0, 327), (348, 675), (696, 1023)]):
        rows = np.where(analyzer[:, 640] == k)[0]
        assert first <= rows[0] <= first + 5
        assert last - 5 <= rows[-1] <= last
        # Elevation increases by one pixel field of view per row
        np.testing.assert_allclose(np.diff(elevation[rows, 640]), ifov, rtol=0.01)
        # Lines of constant elevation curve: the ends of a row look 0.015 degrees
        # (about 3.7 pixels) lower
        centre = rows[len(rows) // 2]
        ends = 0.5 * (elevation[centre, 0] + elevation[centre, 1279])
        assert (elevation[centre, 640] - ends) / ifov == pytest.approx(3.69, abs=0.05)

    # 5.2 degrees across track, the same in every sub-image
    np.testing.assert_allclose(
        np.nanmax(azimuth, axis=1)[[163, 511, 859]], 2.6, atol=0.01
    )
    # The etendue of a fully lit pixel is the effective aperture area times its solid
    # angle, within the pixel response nonuniformity
    etendue = cal_db["pixel_etendue"].to_numpy()[400:600, 300:900]
    np.testing.assert_allclose(
        np.median(etendue), 1.25e-4 * np.deg2rad(ifov) ** 2, rtol=0.01
    )
