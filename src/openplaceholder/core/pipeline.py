from dataclasses import dataclass

from openplaceholder.core.assembly.mapper import Mapper
from openplaceholder.core.assembly.transformation import Transformation
from openplaceholder.core.generation.generator import StructureGenerator
from openplaceholder.core.selection.normalizer import Normalizer
from openplaceholder.core.selection.selector import Selector
from openplaceholder.core.selection.validator import Validator


@dataclass(frozen=True)
class Pipeline:
    generator: StructureGenerator
    validators: list[Validator]
    normalizers: list[Normalizer]
    selector: Selector
    transformations: list[Transformation]
    mapping: Mapper
