import logging
from abc import ABC, abstractmethod

from gufe import AlchemicalNetwork

from openplaceholder.core.configuration import ConfigBase
from openplaceholder.core.interface import Module
from openplaceholder.core.structure import StructureSet

logger = logging.getLogger(__name__)


class MapperConfigBase(ConfigBase): ...


class Mapper(Module, ABC):

    _input_type = StructureSet
    _output_type = AlchemicalNetwork

    @abstractmethod
    def _map(self, structures: StructureSet) -> AlchemicalNetwork:
        raise NotImplementedError

    def map(self, structures: StructureSet) -> AlchemicalNetwork:
        logger.info("mapping structures using %s", self.__class__.__name__)
        return self._map(structures)
