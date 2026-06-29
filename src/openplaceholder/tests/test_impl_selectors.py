from unittest import TestCase

from openplaceholder.impl.selector.mpo import MPOSelector, MPOSelectorConfig


class TestMPOSelector(TestCase):

    def test_init(self) -> None:
        config = MPOSelectorConfig(objectives={})
        MPOSelector(config)
