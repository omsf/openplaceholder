from abc import ABC, abstractmethod

from openplaceholder.core.configuration import ConfigBase
from openplaceholder.core.interface import Module
from openplaceholder.core.structure.structure import Structure, StructureSet


class SelectorConfigBase(ConfigBase): ...


class Selector(Module, ABC):

    @abstractmethod
    def _select(self, structures: list[StructureSet]) -> list[Structure]:
        raise NotImplementedError

    def select(self, structures: list[StructureSet]) -> list[Structure]:
        return self._select(structures)
