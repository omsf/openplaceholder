from unittest import TestCase

from openplaceholder.impl.selectors import (
    CoordinationSelector,
    CoordinationSelectorConfig,
)


class TestCoordinationSelector(TestCase):

    def test_init(self) -> None:
        config = CoordinationSelectorConfig()
        CoordinationSelector(config)
