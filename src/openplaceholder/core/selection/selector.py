from abc import ABC, abstractmethod

from openplaceholder.core.structure.structure import Structure, StructureSet


class Selector(ABC):

    @abstractmethod
    def _select(self, structures: list[StructureSet]) -> list[Structure]:
        raise NotImplementedError

    def select(self, structures: list[StructureSet]) -> list[Structure]:
        return self._select(structures)
