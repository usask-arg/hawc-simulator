from __future__ import annotations

from dataclasses import dataclass

from .base import Data


@dataclass
class Atmosphere(Data):
    constituents: (
        dict  # A dictionary of sasktran2.Constituent objects, with {key: Constituent}
    )
