from unittest import TestCase

import numpy as np

from openplaceholder.core.structure import StructureSet
from openplaceholder.impl.selector.mpo import MPOSelector, MPOSelectorConfig
from openplaceholder.tests.helpers import make_structures


class TestMPOSelector(TestCase):

    def test_init(self) -> None:
        config = MPOSelectorConfig(objectives={})
        MPOSelector(config)

    def test_select_raises_when_pool_too_large(self) -> None:
        selector = MPOSelector(MPOSelectorConfig(objectives={}))
        too_many = StructureSet.from_structures(make_structures(MPOSelector._MAX_POOL_SIZE + 1))
        with self.assertRaises(NotImplementedError):
            selector.select([too_many])

    def test_optimize_picks_best_cross_group_pair(self) -> None:
        selector = MPOSelector(MPOSelectorConfig(objectives={}))
        # groups [0, 1] and [2, 3]; cross-group pair (0, 2) scores highest.
        matrix = np.array(
            [
                [1.0, 0.1, 0.9, 0.2],
                [0.1, 1.0, 0.3, 0.8],
                [0.9, 0.3, 1.0, 0.4],
                [0.2, 0.8, 0.4, 1.0],
            ]
        )
        groups = [[0, 1], [2, 3]]
        self.assertEqual(selector._optimize(matrix, groups), [0, 2])
