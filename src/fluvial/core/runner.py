import logging
from typing import Any

from gufe.network import AlchemicalNetwork
from gufe.tokenization import GufeTokenizable

from fluvial.core.analysis.analyzer import Analyzer
from fluvial.core.assembly.mapper import Mapper
from fluvial.core.assembly.transformation import Transformation
from fluvial.core.generation.generator import (
    StructureGenerator,
)
from fluvial.core.pipeline import Pipeline
from fluvial.core.selection.normalizer import Normalizer
from fluvial.core.selection.selector import Selector
from fluvial.core.selection.validator import Validator
from fluvial.core.simulation.simulator import SimulationResults, Simulator
from fluvial.core.structure import StructureSeries, StructureSet

logger = logging.getLogger(__name__)


def _run_generator(generator_plugin: StructureGenerator) -> StructureSet:
    """Run a structure generator and return the resulting StructureSet."""
    logger.info("running generator %s", generator_plugin.__class__.__name__)
    return generator_plugin.run()


def _run_single_validator(validator: Validator, structures: StructureSet) -> StructureSet:
    """Validate structures using a single validator."""
    logger.debug("validating structures with %s", validator.__class__.__name__)
    return validator.validate_structures(structures)


def _run_single_normalizer(normalizer: Normalizer, structures: StructureSet) -> StructureSet:
    """Normalize structures using a single normalizer."""
    logger.debug("normalizing structures with %s", normalizer.__class__.__name__)
    return normalizer.normalize(structures)


def _run_selector(selector: Selector, structures: StructureSet) -> StructureSeries:
    """Select the best structures from a StructureSet using the given selector."""
    logger.info("selecting structures using %s", selector.__class__.__name__)
    return selector.select(structures)


def _run_single_transformation(
    transformation: Transformation,
    structures: StructureSeries,
) -> StructureSeries:
    """Transform a StructureSeries using the given transformation."""
    logger.debug("applying transformation %s", transformation.__class__.__name__)
    return transformation.transform(structures)


def _run_mapper(mapper: Mapper, structures: StructureSeries) -> AlchemicalNetwork:
    """Map a StructureSeries to an AlchemicalNetwork."""
    logger.info("mapping structures using %s", mapper.__class__.__name__)
    return mapper.map(structures)


def run_serial(pipeline: Pipeline, initial_data: Any) -> GufeTokenizable:
    """Naive and simple implementation for running a pipeline.

    This runs a Pipeline through all of its modules, using initial
    data as the first module's input.

    Parameters
    ----------
    pipeline
        The pipeline to be run.
    initial_data
        Input data to the first module. This depends on the type of module.

    Raises
    ------
    TypeError
        When an unrecognized type is found in the pipeline.
    ValueError
        When the pipeline contains no plugins
    """

    data: GufeTokenizable | None = initial_data

    if len(pipeline.plugins) == 0:
        raise ValueError("Pipeline contains no plugins")

    for plugin in pipeline:
        match plugin, data:
            case StructureGenerator(), None:
                logger.info("Generating structures")
                data = plugin.run()
            case Validator(), StructureSet():
                logger.info("applying validator: %s", plugin.__class__.__name__)
                data = plugin.validate_structures(data)
            case Normalizer(), StructureSet():
                logger.info("applying normalizer: %s", plugin.__class__.__name__)
                data = plugin.normalize(data)
            case Selector(), StructureSet():
                logger.info("selecting structure series from structure pool using: %s", plugin.__class__.__name__)
                data = plugin.select(data)
            case Transformation(), StructureSeries():
                logger.info("applying transformation: %s", plugin.__class__.__name__)
                data = plugin.transform(data)
            case Mapper(), StructureSeries():
                data = plugin.map(data)
            case Simulator(), AlchemicalNetwork():
                data = plugin.simulate(data)
            case Analyzer(), SimulationResults():
                data = plugin.analyze(data)
            case _:
                raise TypeError(f"Unrecognized module {plugin}")
    return data
