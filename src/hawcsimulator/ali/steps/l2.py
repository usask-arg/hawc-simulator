from __future__ import annotations

from aliprocessing.processing.l1b_to_l2 import process_l1b_to_l2_image

from hawcsimulator.steps import Step


class ALIL1bToL2(Step):
    def _run(self, data: dict, cfg: dict) -> dict:  # noqa: ARG002
        data["l2"] = process_l1b_to_l2_image(
            data["l1b"], data["por"].isel(time=0), data["calibration_database"], **cfg
        )

        return data

    def _validate_data(self, data: dict):
        assert "l1b" in data
        assert "por" in data
        assert "calibration_database" in data
