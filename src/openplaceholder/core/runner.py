import logging

from gufe import AlchemicalNetwork

from openplaceholder.core.pipeline import Pipeline
from openplaceholder.core.structure import Structure, StructureSet

logger = logging.getLogger(__name__)


def run_serial(pipeline: Pipeline) -> AlchemicalNetwork:
    """Naive and simple implementation for running a pipeline.

    This differs from an iterative approach in that a pipeline must
    have all types validated during construction.
    """
    generator = pipeline.generator
    selector = pipeline.selector
    mapper = pipeline.mapping
    transformations = pipeline.transformations

    structure_sets: list[StructureSet] = []

    logger.info("Generating structures")
    artifacts = generator.run()

    for artifact in artifacts:
        structures: list[Structure] = artifact.structures

        logger.info(f"Performing validation for {artifact.ligand_name}")

        for validator in pipeline.validators:
            logger.info("applying validator: %s", validator.__class__.__name__)
            structures = validator.validate_structures(structures)

        if not structures:
            logger.warning(f"No structures for ligand {artifact.ligand_name} passed validation, dropping")
            continue
        structure_sets.append(StructureSet.from_structures(structures))

    # selector.select optimizes jointly across all ligands' candidate sets
    # (e.g. cross-ligand pairwise objectives), so it's called once on the
    # full collection rather than per-ligand.
    selected_structures = selector.select(structure_sets)

    logger.info("applying transformations")
    for transformation in transformations:
        logger.info("applying transformation: %s", transformation.__class__.__name__)
        selected_structures = transformation.transform(selected_structures)

    # TODO: technically not the last step, but will leave this here for now
    return mapper.map(selected_structures)
