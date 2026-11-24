from __future__ import annotations

import warnings

from hamilton import registry

import hawcsimulator.steps.atmosphere as atmosphere
import hawcsimulator.steps.limb_observation as limb_observation

registry.disable_autoload()
from hamilton import driver  # noqa: E402


class Simulator:
    def __init__(self) -> None:
        self._modules = [atmosphere, limb_observation]

    def _initialize_data(self) -> dict:
        return {}

    def run(
        self,
        outputs: list[str],
        input: None | dict = None,
        overrides: None | dict = None,
        extra_modules: list | None = None,
        config: dict | None = None,
    ) -> dict:
        warnings.filterwarnings("ignore")

        if input is None:
            input = {}

        if config is None:
            config = {}

        if overrides is None:
            overrides = {}

        default_config = {
            "atmosphere_method": "default",
            "observation_method": "limb",
            "FER_provided": "front_end_radiance" in input,
        }

        default_config.update(config)

        user_input = set(input)
        input = {**self._initialize_data(), **input}

        if extra_modules is not None:
            all_modules = self._modules + extra_modules
        else:
            all_modules = self._modules

        dr = (
            driver.Builder()
            .with_modules(*all_modules)
            .with_config(default_config)
            .build()
        )

        _check_inputs(dr, user_input)

        return dr.execute(outputs, inputs=input, overrides=overrides)


def _check_inputs(dr: driver.Driver, user_input: set[str]) -> None:
    """Reject inputs the simulator would otherwise silently ignore.

    Hamilton ignores an input that names a computed node, and an input that no node
    uses.
    """
    variables = dr.list_available_variables()
    computed = {v.name for v in variables if not v.is_external_input}
    external = {v.name for v in variables if v.is_external_input}

    shadowed = sorted(user_input & computed)
    if shadowed:
        msg = (
            f"{shadowed} are computed by this simulator, so passing them as input has "
            "no effect; pass them in `overrides` to replace the computed values"
        )
        raise ValueError(msg)

    unused = sorted(user_input - computed - external)
    if unused:
        msg = f"{unused} are not used by this simulator with this configuration"
        raise ValueError(msg)
