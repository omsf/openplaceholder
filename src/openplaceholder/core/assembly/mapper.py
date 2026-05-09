from abc import ABC, abstractmethod

from gufe import AlchemicalNetwork

from openplaceholder.core.structure.structure import Structure


class Mapper(ABC):

    @abstractmethod
    def map(self, structures: list[Structure]) -> AlchemicalNetwork:
        raise NotImplementedError
