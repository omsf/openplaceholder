from dataclasses import dataclass

import numpy as np

from openplaceholder.core.selection.selector import Selector
from openplaceholder.core.structure import Structure, StructureSet
from openplaceholder.impl.selector.objective import Objective


@dataclass(frozen=True, eq=True)
class MPOSelectorConfig:
    pass


class MPOSelector(Selector):
    """Multi-parameter optimization selector.

    Flattens the per-ligand candidate sets into a single pool, asks each
    objective for its pairwise score matrix over that pool, combines those
    matrices into one weighted matrix, then optimizes that matrix to pick a
    single structure per ligand.
    """

    def __init__(self, config: MPOSelectorConfig):
        self._config = config
        # (objective, weight) pairs; populated from config once wiring lands.
        self._objectives: list[tuple[Objective, float]] = []

    def select(self, structures: list[StructureSet]) -> list[Structure]:
        pool, groups = self._flatten(structures)
        combined = self._combine(pool)
        chosen = self._optimize(combined, groups)
        return [pool[i] for i in chosen]

    @staticmethod
    def _flatten(
        structures: list[StructureSet],
    ) -> tuple[list[Structure], list[list[int]]]:
        """Flatten per-ligand sets into one pool plus per-ligand index groups.

        ``groups[k]`` holds the pool indices of the candidate structures for
        ligand ``k``; the optimizer must pick exactly one index from each group.
        """
        pool: list[Structure] = []
        groups: list[list[int]] = []
        for structure_set in structures:
            group = []
            for structure in structure_set.structures:
                group.append(len(pool))
                pool.append(structure)
            groups.append(group)
        return pool, groups

    def _combine(self, pool: list[Structure]) -> np.ndarray:
        """Weighted sum of each objective's pairwise matrix over the pool."""
        n = len(pool)
        combined = np.zeros((n, n), dtype=float)
        for objective, weight in self._objectives:
            combined += weight * objective.matrix(pool)
        return combined

    def _optimize(self, matrix: np.ndarray, groups: list[list[int]]) -> list[int]:
        """Pick one structure index per ligand group, maximizing the combined
        pairwise score across the selected combination."""
        raise NotImplementedError
