from unittest import TestCase

from openplaceholder.impl.mappers import LOMAPMapper, LOMAPMapperConfig


class TestLOMAPMapper(TestCase):

    def test_init(self) -> None:
        config = LOMAPMapperConfig()
        LOMAPMapper(config)
