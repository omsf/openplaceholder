from abc import ABC, abstractmethod

from openplaceholder.core.configuration import ConfigBase
from openplaceholder.core.interface import Module
from openplaceholder.core.structure import Structure


class TransformationConfigBase(ConfigBase): ...


class Transformation(Module, ABC):

    @abstractmethod
    def _transform(self, structures: list[Structure]) -> list[Structure]:
        raise NotImplementedError

    def transform(self, structures: list[Structure]) -> list[Structure]:
        return self._transform(structures)
