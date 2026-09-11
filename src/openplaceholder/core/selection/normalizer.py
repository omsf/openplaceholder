import logging
from abc import ABC, abstractmethod

from openplaceholder.core.configuration import ConfigBase
from openplaceholder.core.interface import Module
from openplaceholder.core.structure import StructureSet

logger = logging.getLogger(__name__)


class NormalizerConfigBase(ConfigBase): ...


class Normalizer(Module, ABC):
    """Puts candidate structures on a common footing, before any are chosen.

    Normalizers run over the whole StructureSet, so unlike a ``Transformation``
    they see every replicate rather than the selection made from them, and they
    leave each complex chemically untouched.
    """

    @abstractmethod
    def _normalize(self, structures: StructureSet) -> StructureSet:
        raise NotImplementedError

    def normalize(self, structures: StructureSet) -> StructureSet:
        logger.info("normalizing structures using %s", self.__class__.__name__)
        return self._normalize(structures)
