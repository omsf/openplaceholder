import logging
from abc import ABC, abstractmethod

from openplaceholder.core.configuration import ConfigBase
from openplaceholder.core.interface import Module
from openplaceholder.core.structure import Structure, StructureSet

logger = logging.getLogger(__name__)


class ValidatorConfigBase(ConfigBase): ...


class Validator(Module, ABC):

    def validate_structures(self, structures: StructureSet) -> StructureSet:
        logger.info("validating structures with %s", self.__class__.__name__)
        _structures = []
        for outer in structures.iter_replicates():
            _structures.append(
                [structure for structure in outer.iter_replicates() if self._validate_structure(structure)]
            )
        return StructureSet.from_structures(_structures)

    @abstractmethod
    def _validate_structure(self, structure: Structure) -> bool:
        raise NotImplementedError
