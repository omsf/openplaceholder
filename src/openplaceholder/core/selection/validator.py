import logging
from abc import ABC, abstractmethod

from openplaceholder.core.configuration import ConfigBase
from openplaceholder.core.interface import Module
from openplaceholder.core.structure import (
    EmptyReplicatesError,
    Structure,
    StructureReplicates,
    StructureSet, EmptyStructureSetError,
)

logger = logging.getLogger(__name__)


class ValidatorConfigBase(ConfigBase): ...


class Validator(Module, ABC):

    def validate_structures(self, structures: StructureSet) -> StructureSet:
        """Validate the structures within a StructureSet, removing
        those that do not pass validation.

        Validation is performed on each Structure
        individually. Resulting StructureReplicates will be dropped

        Parameters
        ----------
        structures
            The structures to be validated, contained within a StructureSet.

        Raises
        ------
        EmptyStructureSetError
            When no structures provided were considered valid.
        """
        logger.info("validating structures with %s", self.__class__.__name__)
        _structures = []
        for outer in structures.iter_replicates():
            try:
                _structures.append(
                    StructureReplicates(
                        [structure for structure in outer.iter_replicates() if self._validate_structure(structure)]
                    )
                )
            except EmptyReplicatesError:
                logger.warning(f"No valid structures found for: {outer.ligand_name}")
                continue

        try:
            instance = StructureSet(_structures)
        except EmptyStructureSetError as e:
            logger.error("StructureSet contained no valid structures")
            raise e

        return instance

    @abstractmethod
    def _validate_structure(self, structure: Structure) -> bool:
        """Validation for an individual Structure.

        Returns True when valid and False when invalid.
        """
        raise NotImplementedError
