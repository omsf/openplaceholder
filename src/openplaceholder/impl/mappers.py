from dataclasses import dataclass

from gufe import AlchemicalNetwork

from openplaceholder.core.assembly.mapper import Mapper
from openplaceholder.core.structure.structure import Structure


@dataclass(frozen=True, eq=True)
class LOMAPMapperConfig:
    pass


class LOMAPMapper(Mapper):

    def __init__(self, config: LOMAPMapperConfig):
        self._config = config

    def map(self, structures: list[Structure]) -> AlchemicalNetwork:
        raise NotImplementedError
