from dataclasses import dataclass, field, fields
from itertools import combinations
from typing import Any

import numpy as np
from scipy.optimize import LinearConstraint, milp

import openplaceholder.impl.selector.objectives  # noqa: F401  (populate registry)
from openplaceholder.core.loader import resolve_config_type
from openplaceholder.core.selection.objective import Objective, get_objective
from openplaceholder.core.selection.selector import Selector, SelectorConfigBase
from openplaceholder.core.structure import Structure, StructureSet


@dataclass(frozen=True)
class MPOSelectorConfig(SelectorConfigBase):
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

    _config: MPOSelectorConfig

    def _setup(self) -> None:
        self._objectives: list[tuple[Objective, float]] = [
            (self._build_objective(name, settings), float(settings.get("weight", 1.0)))
            for name, settings in self._config.objectives.items()
        ]

    @staticmethod
    def _build_objective(name: str, settings: dict[str, Any]) -> Objective:
        cls = get_objective(name)
        config_type = resolve_config_type(cls)
        params = {k: v for k, v in settings.items() if k != "weight"}
        return cls(config=config_type(**params))

    # _optimize is a MILP with O(n^2) binary variables; past this pool size it
    # routinely fails to converge within a reasonable time (benchmarked: solves
    # in ~1s up to n=36, but n=40+ can take 20s+ depending on how tied the
    # candidates' scores are, especially with >3 candidates per ligand).
    _MAX_POOL_SIZE = 40

    def _select(self, structures: list[StructureSet]) -> list[Structure]:
        pool, groups = self._flatten(structures)
        if len(pool) > self._MAX_POOL_SIZE:
            raise NotImplementedError(
                f"MPOSelector cannot optimize over {len(pool)} candidate structures "
                f"(limit: {self._MAX_POOL_SIZE}); the MILP becomes intractably slow beyond this size."
            )
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
        pairwise score across the selected combination.

        This is a quadratic assignment problem, so it's posed as a MILP: one
        binary "chosen" variable per structure, plus one per pair encoding
        "both endpoints chosen" so the pairwise scores can enter the
        objective linearly.
        """
        n = matrix.shape[0]
        pairs = list(combinations(range(n), 2))

        objective = self._pair_objective(matrix, pairs, n)
        constraints = [
            self._one_per_group_constraint(groups, n, len(pairs)),
            self._pair_linearization_constraints(pairs, n),
        ]

        result = milp(objective, constraints=constraints, integrality=1, bounds=(0, 1))
        if not result.success:
            raise RuntimeError(f"selection optimization failed: {result.message}")

        return np.flatnonzero(np.round(result.x[:n])).tolist()

    @staticmethod
    def _pair_objective(matrix: np.ndarray, pairs: list[tuple[int, int]], n: int) -> np.ndarray:
        """milp minimizes, so negate the pairwise scores to maximize them."""
        c = np.zeros(n + len(pairs))
        for k, (i, j) in enumerate(pairs):
            c[n + k] = -matrix[i, j]
        return c

    @staticmethod
    def _one_per_group_constraint(groups: list[list[int]], n: int, n_pairs: int) -> LinearConstraint:
        """Exactly one structure chosen per ligand group."""
        rows = np.zeros((len(groups), n + n_pairs))
        for g, group in enumerate(groups):
            rows[g, group] = 1
        return LinearConstraint(rows, lb=1, ub=1)

    @staticmethod
    def _pair_linearization_constraints(pairs: list[tuple[int, int]], n: int) -> LinearConstraint:
        """Enforce y_ij == x_i AND x_j for binary x_i, x_j via:
        y_ij <= x_i, y_ij <= x_j, y_ij >= x_i + x_j - 1.
        """
        rows = np.zeros((3 * len(pairs), n + len(pairs)))
        ub = np.zeros(3 * len(pairs))
        for k, (i, j) in enumerate(pairs):
            y = n + k
            rows[3 * k, [y, i]] = [1, -1]
            rows[3 * k + 1, [y, j]] = [1, -1]
            rows[3 * k + 2, [y, i, j]] = [-1, 1, 1]
            ub[3 * k + 2] = 1
        return LinearConstraint(rows, lb=-np.inf, ub=ub)
