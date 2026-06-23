from dataclasses import dataclass

from openplaceholder.core.selection.selector import Selector, SelectorConfigBase
from openplaceholder.core.structure import Structure, StructureSet


@dataclass(frozen=True, eq=True)
class CoordinationSelectorConfig(SelectorConfigBase):
    pass


class CoordinationSelector(Selector):

    _config: CoordinationSelectorConfig

    def _setup(self) -> None:
        pass

    def _select(self, structures: list[StructureSet]) -> list[Structure]:
        raise NotImplementedError
