from abc import ABC, abstractmethod

from openplaceholder.core.structure.structure import Structure


class Filter(ABC):

    @abstractmethod
    def filter(self, structures: list[Structure]) -> list[Structure]:
        raise NotImplementedError
