from dataclasses import dataclass

from openplaceholder.core.assembly.mapper import Mapper
from openplaceholder.core.assembly.transformation import Transformation
from openplaceholder.core.generation.generator import StructureGenerator
from openplaceholder.core.selection.selector import Selector
from openplaceholder.core.selection.validator import Validator


@dataclass(frozen=True, eq=True)
class Pipeline:
    generator: StructureGenerator
    validators: list[Validator]
    selector: Selector
    # TODO: layered transformations? these might be order dependent so
    # need to be careful
    transformation: Transformation | None
    mapping: Mapper
