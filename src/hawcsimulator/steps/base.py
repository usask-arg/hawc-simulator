from __future__ import annotations

import abc


class Step(abc.ABC):
    @abc.abstractmethod
    def _run(self, data: dict, cfg: dict) -> dict:
        pass

    def _validate_data(self, data: dict):
        assert data is not None

    def _validate_cfg(self, cfg: dict):
        assert cfg is not None

    def run(self, data: dict, cfg: dict) -> dict:
        self._validate_data(data)
        self._validate_cfg(cfg)
        return self._run(data, cfg)
