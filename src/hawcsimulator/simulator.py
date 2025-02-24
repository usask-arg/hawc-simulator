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
        pass

    def run(
        self,
        outputs: list[str],
        input: None | dict = None,
        extra_modules: list | None = None,
        config: dict | None = None,
    ) -> dict:
        warnings.filterwarnings("ignore")

        if input is None:
            input = {}

        if config is None:
            config = {}

        default_config = {"atmosphere_method": "default", "observation_method": "limb"}

        default_config.update(config)

        input = {**self._initialize_data(), **input}

        if extra_modules is not None:
            all_modules = self._modules + extra_modules
        else:
            all_modules = self._modules

        dr = (
            driver.Builder()
            .with_config(
                {}
            )  # we don't have any configuration or invariant data for this example.
            .with_modules(
                *all_modules
            )  # we need to tell hamilton where to load function definitions from
            .with_config(default_config)
            .build()
        )

        return dr.execute(outputs, inputs=input)


if __name__ == "__main__":
    test = Simulator()

    res = test.run(input={"constituents": {"rayleigh": None}})
