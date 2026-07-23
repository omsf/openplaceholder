from unittest import TestCase

from openplaceholder.impl.mappers import (
    KartografMapper,
    KartografMapperConfig,
    LOMAPMapper,
    LOMAPMapperConfig,
)


class TestLOMAPMapper(TestCase):

    def test_init(self) -> None:
        config = LOMAPMapperConfig()
        LOMAPMapper(config)


class TestKartographMapper(TestCase):

    def test_init(self) -> None:
        config = KartografMapperConfig()
        KartografMapper(config)
