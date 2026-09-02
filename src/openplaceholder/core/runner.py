import logging
from typing import Any

from gufe.tokenization import GufeTokenizable

from openplaceholder.core.assembly.mapper import Mapper
from openplaceholder.core.assembly.transformation import Transformation
from openplaceholder.core.generation.generator import (
    StructureGenerator,
)
from openplaceholder.core.pipeline import Pipeline
from openplaceholder.core.selection.selector import Selector
from openplaceholder.core.selection.validator import Validator

logger = logging.getLogger(__name__)


def run_serial(pipeline: Pipeline, initial_data: Any) -> GufeTokenizable:
    """Naive and simple implementation for running a pipeline.

    This differs from an iterative approach in that a pipeline must
    have all types validated during construction.
    """

    data = initial_data
    for plugin in pipeline:
        match plugin:
            case StructureGenerator():
                logger.info("Generating structures")
                data = plugin.run()
            case Validator():
                logger.info("applying validator: %s", plugin.__class__.__name__)
                data = plugin.validate_structures(data)
            case Selector():
                data = plugin.select(data)
            case Transformation():
                logger.info("applying transformation: %s", plugin.__class__.__name__)
                data = plugin.transform(data)
            case Mapper():
                data = plugin.map(data)
            case _:
                raise TypeError(f"Unrecognized module {plugin}")
    return data
