from dataclasses import dataclass

from openplaceholder.core.assembly.transformation import Transformation
from openplaceholder.core.structure.structure import Structure


@dataclass(frozen=True, eq=True)
class MaxVolumeSiteTransformationConfig:
    pass


class MaxVolumeSiteTransformation(Transformation):

    def __init__(self, config: MaxVolumeSiteTransformationConfig):
        self._config = config

    def transform(self, structures: list[Structure]) -> list[Structure]:
        # TODO: this should reflect what was done in the openfe example notebook
        raise NotImplementedError
