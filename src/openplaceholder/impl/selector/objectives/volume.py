"""Shared-volume objective: rewards pairs of ligand poses that occupy the
same region of 3D space."""

from dataclasses import dataclass

from openplaceholder.core.selection.objective import Objective, ObjectiveConfig
from openplaceholder.core.structure import Structure


@dataclass(frozen=True, eq=True)
class VolumeOverlapObjectiveConfig(ObjectiveConfig):
    pass


class VolumeOverlapObjective(Objective):

    _config: VolumeOverlapObjectiveConfig

    def score(self, a: Structure, b: Structure) -> float:
        raise NotImplementedError
