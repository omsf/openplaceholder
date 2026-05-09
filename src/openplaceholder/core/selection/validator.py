from abc import ABC, abstractmethod

from openplaceholder.core.structure.structure import Structure


class Validator(ABC):

    @abstractmethod
    def validate(self, structures: list[Structure]) -> list[Structure]:
        raise NotImplementedError
