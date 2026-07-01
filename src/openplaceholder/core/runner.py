from gufe import AlchemicalNetwork

from openplaceholder.core.assembly.mapper import Mapper
from openplaceholder.core.assembly.transformation import Transformation
from openplaceholder.core.generation.generator import StructureGenerator
from openplaceholder.core.pipeline import Pipeline
from openplaceholder.core.selection.selector import Selector
from openplaceholder.core.selection.validator import Validator
from openplaceholder.core.structure import Structure, StructureSet


def run_serial(pipeline: Pipeline) -> AlchemicalNetwork:
    """Naive and simple implementation for running a pipeline.

    This differs from an iterative approach in that a pipeline must
    have all types validated during construction.
    """
    generator = pipeline.generator
    selector = pipeline.selector
    mapper = pipeline.mapping
    transformation = pipeline.transformation

    structure_sets: list[StructureSet] = []
    for artifact in generator.run():
        structures: list[Structure] = artifact.structures

        for validator in pipeline.validators:
            structures = validator.validate_structures(structures)

        structure_sets.append(StructureSet.from_structures(structures))

    # selector.select optimizes jointly across all ligands' candidate sets
    # (e.g. cross-ligand pairwise objectives), so it's called once on the
    # full collection rather than per-ligand.
    selected_structures = selector.select(structure_sets)

    if transformation is not None:
        selected_structures = transformation.transform(selected_structures)

    # TODO: technically not the last step, but will leave this here for now
    return mapper.map(selected_structures)
