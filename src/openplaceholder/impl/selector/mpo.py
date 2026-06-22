from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

import openplaceholder.impl.selector.objectives  # noqa: F401  (populate registry)
from openplaceholder.core.loader import resolve_config_type
from openplaceholder.core.selection.objective import Objective, get_objective
from openplaceholder.core.selection.selector import Selector
from openplaceholder.core.structure import Structure, StructureSet


@dataclass(frozen=True, eq=True)
class MPOSelectorConfig:
    # objectives keyed by class name, e.g. {"VolumeOverlapObjective": {"weight": 1.0}}.
    # Each entry's "weight" is its weight in the combination; remaining keys are
    # passed to the objective's config.
    objectives: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, settings in self.objectives.items():
            cls = get_objective(name)  # raises on unknown objective
            if not isinstance(settings, dict):
                raise TypeError(f"objective '{name}' settings must be a table")
            params = {k: v for k, v in settings.items() if k != "weight"}
            valid = {f.name for f in fields(resolve_config_type(cls))}
            if extra := set(params) - valid:
                raise ValueError(f"Unknown settings for objective '{name}': {extra}")


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
            (self._build_objective(name, settings), float(settings.get("weight", 1.0)))
            for name, settings in config.objectives.items()
        ]

    @staticmethod
    def _build_objective(name: str, settings: dict[str, Any]) -> Objective:
        cls = get_objective(name)
        config_type = resolve_config_type(cls)
        params = {k: v for k, v in settings.items() if k != "weight"}
        return cls(config=config_type(**params))

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
