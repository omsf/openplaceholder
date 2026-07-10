from unittest import TestCase

from openplaceholder.impl.mappers import LOMAPMapper, LOMAPMapperConfig
from openplaceholder.impl.mappers import KartografMapper, KartografMapperConfig


class TestLOMAPMapper(TestCase):

    def test_init(self) -> None:
        config = LOMAPMapperConfig()
        LOMAPMapper(config)

class TestKartographMapper(TestCase):

    def test_init(self) -> None:
        config = KartografMapperConfig()
        mapper = KartografMapper(config)
        assert mapper._config.central_ligand is None
