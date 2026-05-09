from dataclasses import dataclass

from openplaceholder.core.selection.filter import Filter
from openplaceholder.core.structure.structure import Structure


@dataclass(frozen=True, eq=True)
class PROLIFFilterConfig:
    # TODO: not a real metric, will figure this out
    minimum_fraction: float = 0.6


class PROLIFFilter(Filter):

    def __init__(self, config: PROLIFFilterConfig):
        self._config = config

    def filter(self, structures: list[Structure]) -> list[Structure]:
        raise NotImplementedError
