from abc import ABC, abstractmethod

from openplaceholder.core.structure.structure import Structure


class Selector(ABC):

    @abstractmethod
    def select(self, structures: list[Structure]) -> Structure:
        raise NotImplementedError
