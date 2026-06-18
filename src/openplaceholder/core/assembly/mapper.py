from abc import ABC, abstractmethod

from gufe import AlchemicalNetwork

from openplaceholder.core.structure.structure import Structure


class Mapper(ABC):

    @abstractmethod
    def _map(self, structures: list[Structure]) -> AlchemicalNetwork:
        raise NotImplementedError

    def map(self, structures: list[Structure]) -> AlchemicalNetwork:
        return self._map(structures)
