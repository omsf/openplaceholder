from abc import ABC, abstractmethod

from gufe import AlchemicalNetwork

from openplaceholder.core.configuration import ConfigBase
from openplaceholder.core.interface import Module
from openplaceholder.core.structure import Structure


class MapperConfigBase(ConfigBase): ...


class Mapper(Module, ABC):

    @abstractmethod
    def _map(self, structures: list[Structure]) -> AlchemicalNetwork:
        raise NotImplementedError

    def map(self, structures: list[Structure]) -> AlchemicalNetwork:
        return self._map(structures)
