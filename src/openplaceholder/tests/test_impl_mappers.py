from openplaceholder.impl.mappers import (
    KartografMapper,
    KartografMapperConfig,
    LOMAPMapper,
    LOMAPMapperConfig,
)


class TestLOMAPMapper:

    def test_init(self) -> None:
        config = LOMAPMapperConfig()
        LOMAPMapper(config)


class TestKartographMapper:

    def test_init(self) -> None:
        config = KartografMapperConfig()
        KartografMapper(config)
