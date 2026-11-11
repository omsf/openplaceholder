import logging
from abc import ABC, abstractmethod

from fluvial.core.configuration import ConfigBase
from fluvial.core.interface import Module
from fluvial.core.structure import StructureSeries, StructureSet

logger = logging.getLogger(__name__)


class SelectorConfigBase(ConfigBase): ...


class Selector(Module, ABC):

    @abstractmethod
    def _select(self, structures: StructureSet) -> StructureSeries:
        raise NotImplementedError

    def select(self, structures: StructureSet) -> StructureSeries:
        logger.info("selecting structures for structure sets using %s", self.__class__.__name__)
        return self._select(structures)
