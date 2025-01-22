from __future__ import annotations

from hawcsimulator.ali.por import por_from_atmosphere
from hawcsimulator.steps import Step


class ALIPORFromAtmosphere(Step):
    def _run(self, data: dict, cfg: dict) -> dict:  # noqa: ARG002
        data["por"] = por_from_atmosphere(
            data["sk_atmosphere"], data["observation_time"]
        )

        return data

    def _validate_data(self, data: dict):
        assert "sk_atmosphere" in data
        assert "observation_time" in data
