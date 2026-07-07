import logging
from abc import ABC, abstractmethod

from gufe import AlchemicalNetwork

from openplaceholder.core.configuration import ConfigBase
from openplaceholder.core.interface import Module
from openplaceholder.core.structure import Structure

logger = logging.getLogger(__name__)


class MapperConfigBase(ConfigBase): ...


class Mapper(Module, ABC):

    @abstractmethod
    def _map(self, structures: list[Structure]) -> AlchemicalNetwork:
        raise NotImplementedError

    def map(self, structures: list[Structure]) -> AlchemicalNetwork:
        logger.info("mapping structures using %s", self.__class__.__name__)
        return self._map(structures)
