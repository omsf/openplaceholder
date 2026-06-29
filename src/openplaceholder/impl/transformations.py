from dataclasses import dataclass

from openplaceholder.core.assembly.transformation import (
    Transformation,
    TransformationConfigBase,
)
from openplaceholder.core.structure import Structure


@dataclass(frozen=True)
class MaxVolumeSiteTransformationConfig(TransformationConfigBase):
    pass


class MaxVolumeSiteTransformation(Transformation):

    _config: MaxVolumeSiteTransformationConfig

    def _setup(self) -> None:
        pass

    def _transform(self, structures: list[Structure]) -> list[Structure]:
        # TODO: this should reflect what was done in the openfe example notebook
        raise NotImplementedError
