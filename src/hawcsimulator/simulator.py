from __future__ import annotations

import warnings

import hawcsimulator.steps as sim_steps
from hawcsimulator.steps import Step


class Simulator:
    def __init__(self, steps: list[Step]) -> None:
        self._steps = steps

        self._steps.append(sim_steps.CreateLimbObservation())
        self._steps.append(sim_steps.DefaultAtmosphere())

    def _initialize_data(self, data: dict) -> dict:
        pass

    def run(self, cfg: dict, data: dict | None = None) -> dict:
        warnings.filterwarnings("ignore")

        if data is None:
            data = {}

        data = self._initialize_data(data)

        for k, v in cfg.get("data", {}).items():
            data[k] = v

        for step_name in cfg["steps"]:
            step_cfg = cfg.get(step_name, {})
            step = next(
                step
                for step in self._steps
                if step_name.lower().replace("_", "")
                in step.__class__.__name__.lower().replace("_", "")
            )
            data = step.run(data, step_cfg)
        return data


if __name__ == "__main__":
    cfg = {"steps": ["generate_l1b", "l1b_to_l2"], "generate_l1b": {}, "l1b_to_l2": {}}
