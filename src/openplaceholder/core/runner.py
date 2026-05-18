from typing import cast

from gufe import AlchemicalNetwork

from openplaceholder.core.assembly.mapper import Mapper
from openplaceholder.core.assembly.transformation import Transformation
from openplaceholder.core.generation.generator import StructureGenerator
from openplaceholder.core.pipeline import Pipeline
from openplaceholder.core.selection.filter import Filter
from openplaceholder.core.selection.selector import Selector
from openplaceholder.core.selection.validator import Validator
from openplaceholder.core.structure.structure import Structure


def run_serial(pipeline: Pipeline) -> AlchemicalNetwork:
    """Naive and simple implementation for running a pipeline.

    This differs from an iterative approach in that a pipeline must
    have all types validated during construction.
    """
    generator = cast(StructureGenerator, pipeline.generator.instance)
    selector = cast(Selector, pipeline.selector.instance)
    mapper = cast(Mapper, pipeline.mapping.instance)
    transformation = cast(Transformation, pipeline.transformation.instance) if pipeline.transformation else None

    selected_structures: list[Structure] = []
    for artifact in generator.run():
        structures: list[Structure] = artifact.structures

        for step in pipeline.validators:
            v = cast(Validator, step.instance)
            structures = v.validate(structures)

        if pipeline.filters is not None:
            for step in pipeline.filters:
                f = cast(Filter, step.instance)
                structures = f.filter(structures)

        selected_structures.append(selector.select(structures))

    if transformation is not None:
        selected_structures = transformation.transform(selected_structures)

    # TODO: technically not the last step, but will leave this here for now
    return mapper.map(selected_structures)
