import logging
from abc import ABC, abstractmethod

from openplaceholder.core.configuration import ConfigBase
from openplaceholder.core.interface import Module
from openplaceholder.core.structure import StructureSeries

logger = logging.getLogger(__name__)


class TransformationConfigBase(ConfigBase): ...


class Transformation(Module, ABC):

    @abstractmethod
    def _transform(self, structures: StructureSeries) -> StructureSeries:
        raise NotImplementedError

    def transform(self, structures: StructureSeries) -> StructureSeries:
        if len(structures) == 0:
            logger.error("%s received and empty list of structures", self.__class__.__name__)
            raise ValueError("no structures provided to transformation")
        logger.info("transforming structures using %s", self.__class__.__name__)
        return self._transform(structures)
