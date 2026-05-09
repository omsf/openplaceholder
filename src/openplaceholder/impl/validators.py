from dataclasses import dataclass

from openplaceholder.core.selection.validator import Validator
from openplaceholder.core.structure.structure import Structure


@dataclass(frozen=True, eq=True)
class PosebustersValidatorConfig:
    pass


class PosebustersValidator(Validator):

    def __init__(self, config: PosebustersValidatorConfig):
        self._config = config

    def validate(self, structures: list[Structure]) -> list[Structure]:
        raise NotImplementedError
