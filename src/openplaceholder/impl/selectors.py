from dataclasses import dataclass

from openplaceholder.core.selection.selector import Selector
from openplaceholder.core.structure.structure import Structure


@dataclass(frozen=True, eq=True)
class FirstSelectorConfig:
    pass


@dataclass(frozen=True, eq=True)
class RandomSelectorConfig:
    pass


@dataclass(frozen=True, eq=True)
class CoordinationSelectorConfig:
    pass


class FirstSelector(Selector):

    def __init__(self, config: FirstSelectorConfig):
        self._config = config

    def select(self, structures: list[Structure]) -> Structure:
        return structures[0]


class RandomSelector(Selector):

    def __init__(self, config: RandomSelectorConfig):
        self._config = config

    def select(self, structures: list[Structure]) -> Structure:
        import random

        return random.choice(structures)


class CoordinationSelector(Selector):

    def __init__(self, config: CoordinationSelectorConfig):
        self._config = config

    def select(self, structures: list[Structure]) -> Structure:
        raise NotImplementedError
