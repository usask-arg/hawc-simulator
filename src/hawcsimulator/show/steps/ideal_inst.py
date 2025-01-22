from __future__ import annotations

from hawcsimulator.show.inst_model import L1bGeneratorILS
from hawcsimulator.steps import Step


class IdealSHOWModelL1b(Step):
    """
    Requires to be previously calculated

    - calibration_database
    - observation
    - fer

    """

    def _run(self, data: dict, cfg: dict) -> dict:  # noqa: ARG002
        l1b_gen = L1bGeneratorILS(data["calibration_database"], data["observation"])
        data["l1b"] = l1b_gen.run(data["fer"])

        return data

    def _validate_data(self, data: dict):
        assert "calibration_database" in data
        assert "observation" in data
        assert "fer" in data
