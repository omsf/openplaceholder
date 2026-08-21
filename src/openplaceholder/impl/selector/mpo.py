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

    # Per-batch candidate limit used by _optimize_batched.  Each batch is solved
    # as one exact MILP; this is kept conservatively small because the solver's
    # tail latency at n≈40 can be 30 s+ (benchmarked), and the batched path
    # runs many solves in sequence so a slow tail compounds.  A group that
    # exceeds _BATCH_SIZE on its own is not sub-divided -- it forms a single
    # batch and relies on _BATCH_TIME_LIMIT to stay bounded.
    _BATCH_SIZE = 24
    _BATCH_TIME_LIMIT = 10.0

    # Safety ceiling for the total pool.  Building the full pairwise matrix is
    # O(n) in expensive per-structure work (after per-structure caching), but
    # the n² pairwise comparisons and solver overhead still grow; this backstop
    # guards against config errors rather than marking a hard algorithmic limit.
    _MAX_POOL_SIZE_BATCHED = 5000

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

    def _select(self, structures: list[StructureSet]) -> StructureSet:
        pool, groups = self._flatten(structures)
        if len(pool) > self._MAX_POOL_SIZE_BATCHED:
            raise NotImplementedError(
                f"MPOSelector received {len(pool)} candidate structures, which exceeds "
                f"the current ceiling of {self._MAX_POOL_SIZE_BATCHED}. Reduce the "
                f"number of candidates per ligand, or raise _MAX_POOL_SIZE_BATCHED if "
                f"the pairwise matrix construction time is acceptable at this scale."
            )
        combined = self._combine(pool)
        chosen = self._optimize_batched(combined, groups)
        return StructureSet.from_structures([pool[i] for i in chosen])

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

    def _optimize_batched(self, matrix: np.ndarray, groups: list[list[int]]) -> list[int]:
        """Fallback for pools too large to optimize jointly (see _MAX_POOL_SIZE).

        Partitions groups into consecutive batches that each fit the exact
        solver, and solves them in sequence: each batch is optimized exactly
        given the previous batches' choices as fixed context (their pairwise
        scores against the current batch enter as a linear bias rather than
        new y-variables, since a fixed candidate's "chosen" state is a
        constant, not something this batch's MILP needs to decide).

        This trades global optimality for tractability -- choices already
        fixed by earlier batches are never revisited -- but keeps every
        individual solve within the size where the exact MILP is fast.
        """
        chosen: list[int] = []
        for batch_groups in self._batch_groups(groups):
            batch_indices = [i for group in batch_groups for i in group]
            local = {pool_index: local_index for local_index, pool_index in enumerate(batch_indices)}

            sub_matrix = matrix[np.ix_(batch_indices, batch_indices)]
            sub_groups = [[local[i] for i in group] for group in batch_groups]
            bias = matrix[np.ix_(batch_indices, chosen)].sum(axis=1) if chosen else None

            picked = self._optimize(sub_matrix, sub_groups, bias, time_limit=self._BATCH_TIME_LIMIT)
            chosen += [batch_indices[i] for i in picked]
        return chosen

    def _batch_groups(self, groups: list[list[int]]) -> list[list[list[int]]]:
        """Partition groups into consecutive batches, each with a total
        candidate count at or under _BATCH_SIZE (a single oversized group
        still gets its own batch, exceeding the limit, rather than being split)."""
        batches: list[list[list[int]]] = []
        batch: list[list[int]] = []
        batch_size = 0
        for group in groups:
            if batch and batch_size + len(group) > self._BATCH_SIZE:
                batches.append(batch)
                batch, batch_size = [], 0
            batch.append(group)
            batch_size += len(group)
        if batch:
            batches.append(batch)
        return batches

    def _optimize(
        self,
        matrix: np.ndarray,
        groups: list[list[int]],
        bias: np.ndarray | None = None,
        time_limit: float | None = None,
    ) -> list[int]:
        """Pick one structure index per ligand group, maximizing the combined
        pairwise score across the selected combination.

        This is a quadratic assignment problem, so it's posed as a MILP: one
        binary "chosen" variable per structure, plus one per pair encoding
        "both endpoints chosen" so the pairwise scores can enter the
        objective linearly.

        ``bias[i]``, if given, is added to candidate ``i``'s linear
        coefficient directly -- used by ``_optimize_batched`` to account for
        pairwise scores against already-fixed choices from earlier batches,
        without needing new y-variables for those (fixed, not chosen) pairs.

        ``time_limit``, if given, caps the solve; milp still returns its best
        feasible incumbent when the limit is hit (just not necessarily
        proven optimal), which ``_optimize_batched`` accepts as a fallback
        rather than treating as a failure.
        """
        n = matrix.shape[0]
        pairs = list(combinations(range(n), 2))

        objective = self._pair_objective(matrix, pairs, n, bias)
        constraints = [
            self._one_per_group_constraint(groups, n, len(pairs)),
            self._pair_linearization_constraints(pairs, n),
        ]
        options = {"time_limit": time_limit} if time_limit is not None else None

        result = milp(objective, constraints=constraints, integrality=1, bounds=(0, 1), options=options)
        if result.x is None:
            raise RuntimeError(f"selection optimization failed: {result.message}")

        return [int(i) for i in np.flatnonzero(np.round(result.x[:n]))]

    @staticmethod
    def _pair_objective(
        matrix: np.ndarray, pairs: list[tuple[int, int]], n: int, bias: np.ndarray | None = None
    ) -> np.ndarray:
        """milp minimizes, so negate the pairwise scores (and bias) to maximize them."""
        c = np.zeros(n + len(pairs))
        if bias is not None:
            c[:n] = -bias
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
