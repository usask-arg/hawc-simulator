"""Named instrument definitions and processing versions.

An instrument definition describes one design of an instrument, for example
``SHOW-0-SHS-REP1``: INSTRUMENT-PHASE-CONCEPT-VERSION, where PHASE is the mission
phase (0, A, B, ...), CONCEPT names the design concept and VERSION is a maturity
label with a revision number. A processing version, for example ``v0.1``, sets the
L1A to L1B and L2 algorithm settings, and an acquisition pattern, for example
``nominal``, sets how the instrument is pointed and exposed. All three are YAML files
shipped in ``hawcsimulator/<instrument>/instruments``, ``.../processing`` and
``.../acquisition``, and a path to a YAML file of the same form can be used in place
of a name.

Published names are not changed; a change to a definition gets a new name.

Other installed packages can add definitions, for example internal instrument versions
that are not public. A package declares an entry point in the
``hawcsimulator.definitions`` group whose value is an importable package laid out like
hawcsimulator's: ``show/instruments``, ``show/processing``, ``show/acquisition``, the
same for ``ali``, and ``detectors``. Its names are available like hawcsimulator's own;
a name defined by more than one package is an error. Each loaded definition records
where it came from under ``_source``, and the outputs record it.
"""

from __future__ import annotations

import math
import os
import re
from importlib import metadata, resources
from pathlib import Path

import yaml

INSTRUMENT_NAME = re.compile(
    r"^(?P<instrument>[A-Z]+)-(?P<phase>[0-9A-Z])-(?P<concept>[A-Z0-9]+)-(?P<version>[A-Z]+[0-9]+)$"
)
PROCESSING_VERSION = re.compile(r"^v[0-9]+\.[0-9]+$")


def load_instrument(package: str, name: str) -> dict:
    """The instrument definition ``name`` of ``package``, or one loaded from a YAML path."""
    definition = _load(package, "instruments", name)
    match = INSTRUMENT_NAME.match(str(definition.get("name", "")))
    if match is None:
        msg = (
            f"Instrument name {definition.get('name')!r} does not follow "
            "INSTRUMENT-PHASE-CONCEPT-VERSION, e.g. SHOW-0-SHS-REP1"
        )
        raise ValueError(msg)
    expected = package.rsplit(".", 1)[-1].upper()
    if match["instrument"] != expected:
        msg = f"{definition['name']} is not a {expected} instrument"
        raise ValueError(msg)
    return definition


def load_processing(package: str, version: str) -> dict:
    """The processing version ``version`` of ``package``, or one loaded from a YAML path."""
    definition = _load(package, "processing", version)
    if PROCESSING_VERSION.match(str(definition.get("version", ""))) is None:
        msg = (
            f"Processing version {definition.get('version')!r} is not of the form v0.1"
        )
        raise ValueError(msg)
    return definition


def load_acquisition(package: str, name: str) -> dict:
    """The acquisition pattern ``name`` of ``package``, or one loaded from a YAML path."""
    definition = _load(package, "acquisition", name)
    if "name" not in definition:
        msg = f"Acquisition pattern {name!r} has no name"
        raise ValueError(msg)
    return definition


def load_detector(name: str) -> dict:
    """A detector description from ``hawcsimulator/detectors``."""
    return _load("hawcsimulator", "detectors", name)


ENTRY_POINT_GROUP = "hawcsimulator.definitions"


def available(package: str, kind: str) -> list[str]:
    """
    The names in ``package/kind`` from hawcsimulator and any installed definition
    packages, e.g. available("hawcsimulator.show", "instruments").
    """
    return sorted({name for _, name, _ in _entries(package, kind)})


def _plugins() -> list[tuple[str, resources.abc.Traversable]]:
    """(source, root folder) of each installed definition package."""
    plugins = []
    for entry_point in metadata.entry_points(group=ENTRY_POINT_GROUP):
        dist = entry_point.dist
        source = (
            f"{dist.name} {dist.version}" if dist is not None else entry_point.value
        )
        plugins.append((source, resources.files(entry_point.value)))
    return plugins


def _entries(
    package: str, kind: str
) -> list[tuple[str, str, resources.abc.Traversable]]:
    """(source, name, file) of every definition of ``kind`` for ``package``."""
    folders = [(_hawcsimulator_source(), resources.files(package).joinpath(kind))]
    # "hawcsimulator.show" -> "show"; "hawcsimulator" (detectors) -> the root
    instrument = package.removeprefix("hawcsimulator").lstrip(".")
    for source, root in _plugins():
        folder = root.joinpath(instrument, kind) if instrument else root.joinpath(kind)
        folders.append((source, folder))
    entries = []
    for source, folder in folders:
        if not folder.is_dir():
            continue
        entries += [
            (source, entry.name.removesuffix(".yaml"), entry)
            for entry in folder.iterdir()
            if entry.name.endswith(".yaml")
        ]
    return entries


def _hawcsimulator_source() -> str:
    try:
        return f"hawcsimulator {metadata.version('hawcsimulator')}"
    except metadata.PackageNotFoundError:
        return "hawcsimulator"


def provenance(
    instrument: dict,
    processing: dict,
    libraries: tuple[str, ...],
    acquisition: dict | None = None,
) -> dict:
    """
    Attributes recording the instrument, processing version, acquisition pattern and
    package versions.
    """
    attrs = {
        "hawcsimulator_instrument": instrument["name"],
        "hawcsimulator_processing": processing["version"],
    }
    if acquisition is not None:
        attrs["hawcsimulator_acquisition"] = acquisition["name"]
    # Where each came from: hawcsimulator, an installed definition package, or a file
    for key, definition in (
        ("instrument", instrument),
        ("processing", processing),
        ("acquisition", acquisition),
    ):
        if definition is not None and "_source" in definition:
            attrs[f"hawcsimulator_{key}_source"] = definition["_source"]
    for package in ("hawcsimulator", *libraries, "skretrieval", "sasktran2"):
        try:
            attrs[f"{package}_version"] = metadata.version(package)
        except metadata.PackageNotFoundError:
            attrs[f"{package}_version"] = "unknown"
    return attrs


def merge(base: dict, overrides: dict | None) -> dict:
    """``base`` with ``overrides`` merged in, recursively for nested dicts."""
    merged = dict(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load(package: str, kind: str, name: str | os.PathLike) -> dict:
    path = Path(name)
    if path.suffix in (".yaml", ".yml") and path.exists():
        definition = yaml.safe_load(path.read_text("utf-8"))
        definition["_source"] = f"file {path.name}"
        return definition

    found = [
        (source, entry)
        for source, entry_name, entry in _entries(package, kind)
        if entry_name == str(name)
    ]
    if not found:
        msg = (
            f"Unknown {kind[:-1] if kind.endswith('s') else kind} {name!r}; "
            f"available: {', '.join(available(package, kind))}, or a path to a YAML file"
        )
        raise ValueError(msg)
    if len(found) > 1:
        msg = f"{name!r} is defined by more than one package: " + ", ".join(
            source for source, _ in found
        )
        raise ValueError(msg)
    source, entry = found[0]
    definition = yaml.safe_load(entry.read_text("utf-8"))
    definition["_source"] = source
    return definition


def max_single_exposure(definition: dict) -> float | None:
    """
    Longest single detector exposure [s] of an instrument definition, or None when
    its exposures are not limited.

    A definition with ``detector: {coadd_exposures: true}`` is limited to its detector
    gain mode's ``max_exposure_s``; longer integrations average several exposures.
    """
    if not definition["detector"].get("coadd_exposures", False):
        return None
    detector = load_detector(definition["detector"]["model"])
    return float(
        detector["gain_modes"][definition["detector"]["gain_mode"]]["max_exposure_s"]
    )


def read_noise(definition: dict) -> float:
    """
    RMS read noise per exposure [electrons] of an instrument definition: its
    ``detector.read_noise_e``, or else the detector gain mode's typical value.
    """
    settings = definition["detector"]
    if settings.get("read_noise_e") is not None:
        return float(settings["read_noise_e"])
    detector = load_detector(settings["model"])
    return float(detector["gain_modes"][settings["gain_mode"]]["read_noise_e"])


def dead_time(definition: dict) -> float:
    """Time between consecutive exposures [s], ``detector.dead_time_s`` (default 0)."""
    return float(definition["detector"].get("dead_time_s", 0.0))


def split_exposure(total_exposure: float, definition: dict) -> tuple[float, int]:
    """
    The single-exposure time [s] and number of exposures that make up
    ``total_exposure`` for an instrument definition: one exposure, or equal
    exposures of at most its detector's longest single exposure.
    """
    max_exposure = max_single_exposure(definition)
    if max_exposure is None:
        return float(total_exposure), 1
    return coadd(total_exposure, max_exposure)


def coadd(total_exposure: float, max_exposure: float) -> tuple[float, int]:
    """
    Equal exposures of at most ``max_exposure`` that add up to ``total_exposure``:
    the single-exposure time [s] and the number of exposures.
    """
    num_exposures = max(1, math.ceil(total_exposure / max_exposure - 1e-9))
    return total_exposure / num_exposures, num_exposures
