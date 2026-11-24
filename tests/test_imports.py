from __future__ import annotations

import importlib
import pkgutil

import pytest

import hawcsimulator

MODULES = sorted(
    module.name
    for module in pkgutil.walk_packages(hawcsimulator.__path__, "hawcsimulator.")
)


@pytest.mark.parametrize("module", MODULES)
def test_import(module):
    # Catches modules that use APIs the companion libraries have removed, e.g.
    # showlib.cal_db was missing from the previously released showlib
    importlib.import_module(module)
