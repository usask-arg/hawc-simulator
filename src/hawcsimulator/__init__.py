from __future__ import annotations

# hawcsimulator.show and hawcsimulator.ali are not imported here: importing one
# registers that instrument's optical properties with skretrieval, which until
# skretrieval keeps per-subclass registries replaces the other instrument's.
from . import fer

__all__ = ["fer"]
