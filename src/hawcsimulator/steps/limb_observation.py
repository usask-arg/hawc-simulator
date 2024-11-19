from __future__ import annotations

import sasktran2 as sk

from hawcsimulator.geometry.observation import SimulatedObservationGeometry
from hawcsimulator.steps import Step


class CreateLimbObservation(Step):
    def _run(self, data: dict, cfg: dict) -> dict:
        tan_alts = data["viewing_tangent_altitudes"]
        obs_time = cfg["time"]
        viewing_geo = sk.viewinggeo.LimbVertical.from_tangent_parameters(
            solar_handler=sk.solar.SolarGeometryHandlerAstropy(),
            tangent_altitudes=tan_alts,
            tangent_latitude=cfg["tangent_latitude"],
            tangent_longitude=cfg["tangent_longitude"],
            time=obs_time,
            observer_altitude=data["observer_altitude"],
            viewing_azimuth=0.0,
        )

        data["observation"] = SimulatedObservationGeometry(
            viewing_geo=viewing_geo,
            sample_wavel=data["sample_wavelengths"],
        )

        data["observation_time"] = obs_time

        return data

    def _validate_data(self, data: dict):
        assert "viewing_tangent_altitudes" in data
        assert "observer_altitude" in data
        assert "sample_wavelengths" in data

    def _validate_cfg(self, cfg: dict):
        assert "tangent_latitude" in cfg
        assert "tangent_longitude" in cfg
        assert "time" in cfg
