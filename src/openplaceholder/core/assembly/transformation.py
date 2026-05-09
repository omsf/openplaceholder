from abc import ABC, abstractmethod

from openplaceholder.core.structure.structure import Structure


class Transformation(ABC):

    @abstractmethod
    def transform(self, structures: list[Structure]) -> list[Structure]:
        raise NotImplementedError
