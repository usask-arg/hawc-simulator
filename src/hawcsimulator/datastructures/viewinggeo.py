from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from skretrieval.retrieval.observation import Observation

from .base import Data


@dataclass
class ObservationContainer(Data):
    observation: Observation  # skretrieval.retrieval.observation.Observation object
    time: pd.Timestamp  # Reference time of the observation
