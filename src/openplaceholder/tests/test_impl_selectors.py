import numpy as np
import pytest

from openplaceholder.core.structure import StructureSet
from openplaceholder.impl.selector.mpo import MPOSelector, MPOSelectorConfig
from openplaceholder.tests.helpers import make_structures


class TestMPOSelector:

    def test_init(self) -> None:
        config = MPOSelectorConfig(objectives={})
        MPOSelector(config)

    def test_select_raises_when_pool_too_large(self) -> None:
        selector = MPOSelector(MPOSelectorConfig(objectives={}))
        too_many = StructureSet.from_structures([make_structures(MPOSelector._MAX_POOL_SIZE_BATCHED + 1)])
        with pytest.raises(NotImplementedError):
            selector.select(too_many)

    def test_optimize_batched_picks_one_per_group_across_batches(self) -> None:
        selector = MPOSelector(MPOSelectorConfig(objectives={}))
        selector._BATCH_SIZE = 4  # cap at 4 candidates/batch -> 2 groups/batch -> 2 batches

        # groups [0,1], [2,3], [4,5], [6,7]; best cross-group pair within
        # batch 1 (groups 0-1) is (0,2), within batch 2 (groups 2-3) is (4,6).
        n = 8
        matrix = np.full((n, n), 0.1)
        np.fill_diagonal(matrix, 1.0)
        matrix[0, 2] = matrix[2, 0] = 0.9
        matrix[4, 6] = matrix[6, 4] = 0.9

        groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
        chosen = selector._optimize_batched(matrix, groups)

        assert sorted(chosen) == [0, 2, 4, 6]

    def test_optimize_picks_best_cross_group_pair(self) -> None:
        selector = MPOSelector(MPOSelectorConfig(objectives={}))
        # 4x4 pairwise compatibility matrix over a flat pool of 4 candidate
        # structures (rows and columns are pool indices 0-3).  Two ligands with
        # two candidates each: ligand 0 → pool indices [0, 1], ligand 1 → [2, 3].
        # Entry [i, j] is the score for selecting both candidate i and candidate j;
        # the highest cross-group score is [0, 2] = 0.9, so the expected selection
        # is pool index 0 for ligand 0 and pool index 2 for ligand 1.
        matrix = np.array(
            [
                [1.0, 0.1, 0.9, 0.2],
                [0.1, 1.0, 0.3, 0.8],
                [0.9, 0.3, 1.0, 0.4],
                [0.2, 0.8, 0.4, 1.0],
            ]
        )
        groups = [[0, 1], [2, 3]]
        assert selector._optimize(matrix, groups) == [0, 2]
