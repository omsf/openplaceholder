"""Objectives score how well *pairs* of candidate structures co-exist.

The MPO selector evaluates each objective over the flat pool of candidate
structures to produce one pairwise score matrix per objective. Those matrices
are combined (weighted) and optimized to choose a single structure per ligand.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from openplaceholder.core.structure import Structure


@dataclass(frozen=True, eq=True)
class ObjectiveConfig:
    pass


class Objective(ABC):

    def __init__(self, config: ObjectiveConfig):
        self._config = config

    @abstractmethod
    def score(self, a: Structure, b: Structure) -> float:
        """Score a single pair of candidate structures.

        Higher is better: the selector maximizes the (weighted) combination of
        pairwise scores across the chosen set of structures.
        """
        raise NotImplementedError

    def matrix(self, structures: Sequence[Structure]) -> np.ndarray:
        """Pairwise score matrix over a flat pool of candidate structures.

        Entry ``[i, j]`` is ``score(structures[i], structures[j])``. Concrete
        objectives may override this with a vectorized implementation.
        """
        n = len(structures)
        scores = np.empty((n, n), dtype=float)
        for i, a in enumerate(structures):
            for j, b in enumerate(structures):
                scores[i, j] = self.score(a, b)
        return scores
