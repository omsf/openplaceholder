from abc import ABC, abstractmethod

from openplaceholder.core.configuration import ConfigBase
from openplaceholder.core.interface import Module
from openplaceholder.core.structure import Structure


class ValidatorConfigBase(ConfigBase): ...


class Validator(Module, ABC):

    def validate_structures(self, structures: list[Structure]) -> list[Structure]:
        _structures = []
        for structure in structures:
            if self._validate_structure(structure):
                _structures.append(structure)
        return _structures

    @abstractmethod
    def _validate_structure(self, structure: Structure) -> bool:
        raise NotImplementedError
