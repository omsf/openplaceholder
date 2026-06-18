from dataclasses import dataclass

import numpy as np

from openplaceholder.core.loader import load_class, resolve_config_type
from openplaceholder.core.selection.selector import Selector
from openplaceholder.core.structure import Structure, StructureSet
from openplaceholder.impl.selector.objective import Objective


@dataclass(frozen=True, eq=True)
class ObjectiveSpec:
    """A single objective to optimize over, plus its weight in the combination."""

    implementation: str
    weight: float = 1.0


@dataclass(frozen=True, eq=True)
class MPOSelectorConfig:
    objectives: tuple[ObjectiveSpec, ...] = ()

    def __post_init__(self) -> None:
        # TOML gives raw dicts; coerce them into ObjectiveSpec instances.
        coerced = tuple(
            spec if isinstance(spec, ObjectiveSpec) else ObjectiveSpec(**spec)
            for spec in self.objectives
        )
        object.__setattr__(self, "objectives", coerced)


class MPOSelector(Selector):
    """Multi-parameter optimization selector.

    Flattens the per-ligand candidate sets into a single pool, asks each
    objective for its pairwise score matrix over that pool, combines those
    matrices into one weighted matrix, then optimizes that matrix to pick a
    single structure per ligand.
    """

    def __init__(self, config: MPOSelectorConfig):
        self._config = config
        self._objectives: list[tuple[Objective, float]] = [
            (self._build_objective(spec.implementation), spec.weight)
            for spec in config.objectives
        ]

    @staticmethod
    def _build_objective(implementation: str) -> Objective:
        cls = load_class(implementation)
        config_type = resolve_config_type(cls)
        # objective configs are param-less for now; when one gains parameters
        # it defines its own __init__ annotation and we pass them through here.
        objective = cls(config=config_type())
        if not isinstance(objective, Objective):
            raise TypeError(f"{implementation} is not an Objective")
        return objective

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
