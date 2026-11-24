from __future__ import annotations

import pytest
import yaml

from hawcsimulator.definitions import (
    INSTRUMENT_NAME,
    available,
    coadd,
    load_acquisition,
    load_detector,
    load_instrument,
    load_processing,
    merge,
    read_noise,
    split_exposure,
)

PACKAGES = ("hawcsimulator.show", "hawcsimulator.ali")


@pytest.mark.parametrize("package", PACKAGES)
def test_shipped_instruments_follow_the_naming_convention(package):
    names = available(package, "instruments")
    assert names

    for name in names:
        definition = load_instrument(package, name)
        # The file is named after the instrument it defines
        assert definition["name"] == name
        assert (
            INSTRUMENT_NAME.match(name)["instrument"] == package.split(".")[-1].upper()
        )
        load_detector(definition["detector"]["model"])


@pytest.mark.parametrize("package", PACKAGES)
def test_shipped_processing_versions_load(package):
    for version in available(package, "processing"):
        definition = load_processing(package, version)
        assert definition["version"] == version
        assert "l1b" in definition
        assert "l2" in definition


@pytest.mark.parametrize("package", PACKAGES)
def test_shipped_acquisition_patterns_load(package):
    names = available(package, "acquisition")
    assert "nominal" in names
    for name in names:
        assert load_acquisition(package, name)["name"] == name


def test_detector_settings():
    definition = load_instrument("hawcsimulator.show", "SHOW-0-SHS-REP1")
    assert read_noise(definition) == 160.0
    definition["detector"]["read_noise_e"] = 100.0
    assert read_noise(definition) == 100.0

    # 2 s in exposures of at most the Owl 1280's 92.5 ms
    exposure, num_exposures = split_exposure(2.0, definition)
    assert num_exposures == 22
    assert exposure * num_exposures == pytest.approx(2.0)
    definition["detector"]["coadd_exposures"] = False
    assert split_exposure(2.0, definition) == (2.0, 1)


def test_unknown_names_list_the_available_ones():
    with pytest.raises(ValueError, match="SHOW-0-SHS-REP1"):
        load_instrument("hawcsimulator.show", "SHOW-0-SHS-PRELIM9")
    with pytest.raises(ValueError, match="v0.1"):
        load_processing("hawcsimulator.show", "v9.9")


def test_instrument_from_a_file(tmp_path):
    definition = load_instrument("hawcsimulator.show", "SHOW-0-SHS-REP1")
    definition["name"] = "SHOW-A-SHS-TEST1"
    path = tmp_path / "test_instrument.yaml"
    path.write_text(yaml.safe_dump(definition))

    assert (
        load_instrument("hawcsimulator.show", str(path))["name"] == "SHOW-A-SHS-TEST1"
    )


def test_simulator_takes_an_instrument_file(tmp_path):
    from hawcsimulator.show import SHOWSimulator

    definition = load_instrument("hawcsimulator.show", "SHOW-0-SHS-REP1")
    definition["name"] = "SHOW-A-SHS-TEST1"
    definition["viewing"]["num_rows"] = 50
    path = tmp_path / "test_instrument.yaml"
    path.write_text(yaml.safe_dump(definition))

    data = SHOWSimulator().run(["calibration_database"], {"instrument": str(path)})

    assert data["calibration_database"].num_rows == 50
    assert (
        data["calibration_database"].ds.attrs["hawcsimulator_instrument"]
        == "SHOW-A-SHS-TEST1"
    )


def test_instrument_names_are_checked(tmp_path):
    definition = load_instrument("hawcsimulator.show", "SHOW-0-SHS-REP1")
    path = tmp_path / "bad.yaml"

    definition["name"] = "SHOW-phase0"
    path.write_text(yaml.safe_dump(definition))
    with pytest.raises(ValueError, match="INSTRUMENT-PHASE-CONCEPT-VERSION"):
        load_instrument("hawcsimulator.show", str(path))

    definition["name"] = "ALI-0-SHS-PRELIM1"
    path.write_text(yaml.safe_dump(definition))
    with pytest.raises(ValueError, match="not a SHOW instrument"):
        load_instrument("hawcsimulator.show", str(path))


def test_merge():
    base = {"engine": {"num_threads": 1, "num_streams": 2}, "a": 1}

    merged = merge(base, {"engine": {"num_threads": 4}, "b": 2})

    assert merged == {"engine": {"num_threads": 4, "num_streams": 2}, "a": 1, "b": 2}
    # The base is unchanged
    assert base["engine"]["num_threads"] == 1


def test_coadd():
    # Equal exposures no longer than the maximum, adding up to the total
    assert coadd(1.0, 0.0925) == pytest.approx((1.0 / 11, 11))
    assert coadd(0.05, 0.0925) == (0.05, 1)
    assert coadd(0.185, 0.0925) == pytest.approx((0.0925, 2))


def _fake_plugin(tmp_path, monkeypatch, name="SHOW-A-SHS-INTERNAL1"):
    """An installed definition package with one SHOW instrument."""
    import hawcsimulator.definitions as definitions

    definition = load_instrument("hawcsimulator.show", "SHOW-0-SHS-REP1")
    definition.pop("_source")
    definition["name"] = name
    folder = tmp_path / "show" / "instruments"
    folder.mkdir(parents=True)
    (folder / f"{name}.yaml").write_text(yaml.safe_dump(definition))
    monkeypatch.setattr(
        definitions, "_plugins", lambda: [("hawcsimulator-internal 0.1.0", tmp_path)]
    )


def test_definition_packages_add_names(tmp_path, monkeypatch):
    from hawcsimulator.definitions import provenance

    _fake_plugin(tmp_path, monkeypatch)

    assert "SHOW-A-SHS-INTERNAL1" in available("hawcsimulator.show", "instruments")
    assert "SHOW-0-SHS-REP1" in available("hawcsimulator.show", "instruments")
    definition = load_instrument("hawcsimulator.show", "SHOW-A-SHS-INTERNAL1")
    assert definition["_source"] == "hawcsimulator-internal 0.1.0"

    # The outputs record where each definition came from
    attrs = provenance(
        definition, load_processing("hawcsimulator.show", "v0.1"), ("showlib",)
    )
    assert attrs["hawcsimulator_instrument_source"] == "hawcsimulator-internal 0.1.0"
    assert attrs["hawcsimulator_processing_source"].startswith("hawcsimulator")


def test_a_name_defined_twice_is_an_error(tmp_path, monkeypatch):
    _fake_plugin(tmp_path, monkeypatch, name="SHOW-0-SHS-REP1")

    with pytest.raises(ValueError, match="more than one package"):
        load_instrument("hawcsimulator.show", "SHOW-0-SHS-REP1")


# Sections that describe a design rather than its performance. Definitions shipped
# with hawcsimulator are public and must not have them; designs belong in an internal
# definition package
DESIGN_SECTIONS = ("interferometer", "aperture", "efficiency", "pixel_geometry")


@pytest.mark.parametrize("package", PACKAGES)
def test_shipped_instruments_are_representative(package):
    import hawcsimulator.definitions as definitions

    for source, name, _ in definitions._entries(package, "instruments"):
        if not source.startswith("hawcsimulator "):
            continue
        definition = load_instrument(package, name)
        assert not set(DESIGN_SECTIONS) & set(definition), name
