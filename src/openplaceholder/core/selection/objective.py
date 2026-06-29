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

_OBJECTIVE_REGISTRY: dict[str, "type[Objective]"] = {}


def get_objective(name: str) -> "type[Objective]":
    """Look up a registered objective class by its name."""
    try:
        return _OBJECTIVE_REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_OBJECTIVE_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown objective '{name}'. Registered: {known}")


@dataclass(frozen=True, eq=True)
class ObjectiveConfig:
    pass


class Objective(ABC):

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        _OBJECTIVE_REGISTRY[cls.__name__] = cls

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

        Entry ``[i, j]`` is ``score(structures[i], structures[j])``. The default
        builder assumes objectives are **symmetric**
        (``score(a, b) == score(b, a)``) and leaves self-pairs on the diagonal at
        ``1.0`` (only one structure per ligand is ever chosen, so self-pairs are
        never both selected). Objectives that are asymmetric should override this.
        """
        n = len(structures)
        scores = np.ones((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                scores[i, j] = scores[j, i] = self.score(structures[i], structures[j])
        return scores
