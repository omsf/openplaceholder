import logging
from abc import ABC, abstractmethod

from openplaceholder.core.configuration import ConfigBase
from openplaceholder.core.interface import Module
from openplaceholder.core.structure import Structure, StructureSet

logger = logging.getLogger(__name__)


class SelectorConfigBase(ConfigBase): ...


class Selector(Module, ABC):

    @abstractmethod
    def _select(self, structures: list[StructureSet]) -> list[Structure]:
        raise NotImplementedError

    def select(self, structures: list[StructureSet]) -> list[Structure]:
        logger.info("selecting structures for structure sets using %s", self.__class__.__name__)
        return self._select(structures)
