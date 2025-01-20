from __future__ import annotations

from hawcsimulator.ali.inst_model import L1bGeneratorILS, L1bGeneratorIdeal
from hawcsimulator.steps import Step


class IdealALIModelL1b(Step):
    """
    Generates L1b data from the Front End Radiance assuming that the only transformation is
    a convolution of the spectral line shape.

    Requires to be previously calculated

    - calibration_database
    - observation
    - fer

    """

    def _run(self, data: dict, cfg: dict) -> dict:  # noqa: ARG002
        l1b_gen = L1bGeneratorIdeal(data["calibration_database"], data["observation"], pol_states=data["polarization_states"], **cfg)
        data["l1b"] = l1b_gen.run(data["fer"])

        return data

    def _validate_data(self, data: dict):
        assert "calibration_database" in data
        assert "observation" in data
        assert "fer" in data
