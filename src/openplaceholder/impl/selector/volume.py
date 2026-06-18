"""Shared-volume objective: rewards pairs of ligand poses that occupy the
same region of 3D space."""

from dataclasses import dataclass

from openplaceholder.core.structure import Structure
from openplaceholder.impl.selector.objective import Objective, ObjectiveConfig


@dataclass(frozen=True, eq=True)
class VolumeOverlapObjectiveConfig(ObjectiveConfig):
    pass


class VolumeOverlapObjective(Objective):

    def score(self, a: Structure, b: Structure) -> float:
        raise NotImplementedError
