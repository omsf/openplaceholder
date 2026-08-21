import logging
from abc import ABC, abstractmethod

from openplaceholder.core.configuration import ConfigBase
from openplaceholder.core.interface import Module
from openplaceholder.core.structure import Structure, StructureSet

logger = logging.getLogger(__name__)


class ValidatorConfigBase(ConfigBase): ...


class Validator(Module, ABC):

    _input_type = StructureSet
    _output_type = StructureSet

    def validate_structures(self, structures: StructureSet) -> StructureSet:
        logger.info("validating structures with %s", self.__class__.__name__)
        _structures = []
        for structure in structures:
            if self._validate_structure(structure):
                _structures.append(structure)
        return StructureSet(_structures)

    @abstractmethod
    def _validate_structure(self, structure: Structure) -> bool:
        raise NotImplementedError
