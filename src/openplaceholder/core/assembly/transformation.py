from abc import ABC, abstractmethod

from openplaceholder.core.structure.structure import Structure


class Transformation(ABC):

    @abstractmethod
    def _transform(self, structures: list[Structure]) -> list[Structure]:
        raise NotImplementedError

    def transform(self, structures: list[Structure]) -> list[Structure]:
        return self._transform(structures)
