from dataclasses import dataclass

from gufe import AlchemicalNetwork

from openplaceholder.core.assembly.mapper import Mapper, MapperConfigBase
from openplaceholder.core.structure.structure import Structure


@dataclass(frozen=True, eq=True)
class LOMAPMapperConfig(MapperConfigBase): ...


class LOMAPMapper(Mapper):

    _config: LOMAPMapperConfig

    def __init__(self, config: LOMAPMapperConfig):
        self._config = config

    def _map(self, structures: list[Structure]) -> AlchemicalNetwork:
        raise NotImplementedError
