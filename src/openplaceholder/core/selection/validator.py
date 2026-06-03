from abc import ABC, abstractmethod

from openplaceholder.core.structure.structure import Structure


class Validator(ABC):

    def validate(self, structures: list[Structure]) -> list[Structure]:
        return list(filter(self._validate_structure, structures))

    @abstractmethod
    def _validate_structure(self, structure: Structure) -> bool:
        raise NotImplementedError
