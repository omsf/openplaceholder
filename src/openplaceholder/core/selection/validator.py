import logging
from abc import ABC, abstractmethod

from openplaceholder.core.configuration import ConfigBase
from openplaceholder.core.interface import Module
from openplaceholder.core.structure import (
    EmptyReplicateError,
    Structure,
    StructureReplicates,
    StructureSet,
)

logger = logging.getLogger(__name__)


class ValidatorConfigBase(ConfigBase): ...


class Validator(Module, ABC):

    def validate_structures(self, structures: StructureSet) -> StructureSet:
        logger.info("validating structures with %s", self.__class__.__name__)
        _structures = []
        for outer in structures.iter_replicates():
            try:
                _structures.append(
                    StructureReplicates(
                        [structure for structure in outer.iter_replicates() if self._validate_structure(structure)]
                    )
                )
            except EmptyReplicateError:
                continue
        return StructureSet(_structures)

    @abstractmethod
    def _validate_structure(self, structure: Structure) -> bool:
        raise NotImplementedError
