import logging
from abc import ABC, abstractmethod

from openplaceholder.core.configuration import ConfigBase
from openplaceholder.core.interface import Module
from openplaceholder.core.structure import StructureSet

logger = logging.getLogger(__name__)


class NormalizerConfigBase(ConfigBase): ...


class Normalizer(Module, ABC):
    """Apply non-chemical changes to Structures within a StructureSet."""

    @abstractmethod
    def _normalize(self, structures: StructureSet) -> StructureSet:
        """The normalization function that must be implemented. The
        implementation must only perform non-chemical transformations
        to the underlying Structures, returning an entirely new StructureSet.
        """
        raise NotImplementedError

    def normalize(self, structures: StructureSet) -> StructureSet:
        logger.info("normalizing structures using %s", self.__class__.__name__)
        return self._normalize(structures)
