import logging
from abc import ABC, abstractmethod

from gufe import AlchemicalNetwork

from openplaceholder.core.configuration import ConfigBase
from openplaceholder.core.interface import Module
from openplaceholder.core.structure import StructureSeries

logger = logging.getLogger(__name__)


class MapperConfigBase(ConfigBase): ...


class Mapper(Module, ABC):

    @abstractmethod
    def _map(self, structures: StructureSeries) -> AlchemicalNetwork:
        raise NotImplementedError

    def map(self, structures: StructureSeries) -> AlchemicalNetwork:
        logger.info("mapping structures using %s", self.__class__.__name__)
        return self._map(structures)
