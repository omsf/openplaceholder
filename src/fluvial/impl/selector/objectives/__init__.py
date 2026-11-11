"""Concrete objectives.

Importing this package imports each objective module, which registers the
objective with the ``Objective`` registry so it can be discovered by name.
"""

from fluvial.impl.selector.objectives.ifp import IFPSimilarityObjective
from fluvial.impl.selector.objectives.volume import VolumeOverlapObjective

__all__ = ["IFPSimilarityObjective", "VolumeOverlapObjective"]
