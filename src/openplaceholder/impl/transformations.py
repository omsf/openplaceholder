from dataclasses import dataclass

from openplaceholder.core.assembly.transformation import (
    Transformation,
    TransformationConfigBase,
)
from openplaceholder.core.structure.structure import Structure


@dataclass(frozen=True, eq=True)
class MaxVolumeSiteTransformationConfig(TransformationConfigBase):
    pass


class MaxVolumeSiteTransformation(Transformation):

    def transform(self, structures: list[Structure]) -> list[Structure]:
        # TODO: this should reflect what was done in the openfe example notebook
        raise NotImplementedError
