import logging
from abc import ABC, abstractmethod

from openplaceholder.core.configuration import ConfigBase
from openplaceholder.core.interface import Module
from openplaceholder.core.structure import StructureSeries, StructureSet

logger = logging.getLogger(__name__)


class SelectorConfigBase(ConfigBase): ...


class Selector(Module, ABC):

    @abstractmethod
    def _select(self, structures: StructureSet) -> StructureSeries:
        raise NotImplementedError

    def select(self, structures: StructureSet) -> StructureSeries:
        logger.info("selecting structures for structure sets using %s", self.__class__.__name__)
        return self._select(structures)
