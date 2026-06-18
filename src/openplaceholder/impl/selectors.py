from dataclasses import dataclass

from openplaceholder.core.selection.selector import Selector
from openplaceholder.core.structure.structure import Structure, StructureSet


@dataclass(frozen=True, eq=True)
class CoordinationSelectorConfig:
    pass


class CoordinationSelector(Selector):

    def __init__(self, config: CoordinationSelectorConfig):
        self._config = config

    def _select(self, structures: list[StructureSet]) -> Structure:
        raise NotImplementedError
