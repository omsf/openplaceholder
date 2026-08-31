import logging
from typing import cast

from gufe import AlchemicalNetwork

from openplaceholder.core.assembly.mapper import Mapper
from openplaceholder.core.assembly.transformation import Transformation
from openplaceholder.core.generation.generator import (
    StructureGenerator,
)
from openplaceholder.core.pipeline import Pipeline
from openplaceholder.core.selection.selector import Selector
from openplaceholder.core.selection.validator import Validator
from openplaceholder.core.structure import Structure, StructureSet

logger = logging.getLogger(__name__)


def run_serial(pipeline: Pipeline) -> AlchemicalNetwork:
    """Naive and simple implementation for running a pipeline.

    This differs from an iterative approach in that a pipeline must
    have all types validated during construction.
    """

    # payload_result_db: dict[str, tuple[Payload, Any]] = {}
    # task_db = OPHTaskStatusDB.from_filename(Path("my_db.db"), overwrite=True)

    def _apply_validator(data: list[StructureSet], validator: Validator) -> list[StructureSet]:
        new = []
        for artifact in data:
            validated_structures = validator.validate_structures(artifact.structures)
            new.append(StructureSet.from_structures(validated_structures))
        return new

    data = None
    for plugin in pipeline:
        match plugin:
            case StructureGenerator():
                logger.info("Generating structures")
                data = plugin.run()
            case Validator():
                logger.info("applying validator: %s", plugin.__class__.__name__)
                data = _apply_validator(data, plugin)  # type: ignore
            case Selector():
                data = plugin.select(cast(list[StructureSet], data))  # type: ignore
            case Transformation():
                logger.info("applying transformation: %s", plugin.__class__.__name__)
                data = plugin.transform(cast(list[Structure], data))  # type: ignore
            case Mapper():
                data = plugin.map(data)  # type: ignore
    return data
