from abc import ABC, abstractmethod

from openplaceholder.core.structure.structure import Structure, StructureSet


class Selector(ABC):

    @abstractmethod
    def select(self, structures: list[StructureSet]) -> list[Structure]:
        raise NotImplementedError
