from dataclasses import dataclass

from gufe import AlchemicalNetwork

from openplaceholder.core.assembly.mapper import Mapper, MapperConfigBase
from openplaceholder.core.structure import Structure


@dataclass(frozen=True)
class LOMAPMapperConfig(MapperConfigBase): ...


class LOMAPMapper(Mapper):

    _config: LOMAPMapperConfig

    def _setup(self) -> None:
        pass

    def _map(self, structures: list[Structure]) -> AlchemicalNetwork:
        raise NotImplementedError

@dataclass(frozen=True)
class KartografMapperConfig(MapperConfigBase):
    central_ligand: str | None = None

class KartografMapper(Mapper):

    _config: KartografMapperConfig

    def _setup(self) -> None:
        pass

    def _map(self, structures: list[Structure]) -> AlchemicalNetwork:
        raise NotImplementedError
