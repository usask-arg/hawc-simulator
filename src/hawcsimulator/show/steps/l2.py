from __future__ import annotations

from showlib.processing.l1b_to_l2 import process_l1b_to_l2

from hawcsimulator.steps import Step


class SHOWLibL1bToL2(Step):
    def _run(self, data: dict, cfg: dict) -> dict:  # noqa: ARG002
        data["l2"] = process_l1b_to_l2(
            data["l1b"], data["por"], data["calibration_database"]
        )

        return data

    def _validate_data(self, data: dict):
        assert "l1b" in data
        assert "por" in data
        assert "calibration_database" in data
