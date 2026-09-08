import logging
from dataclasses import dataclass
from pathlib import Path

from openplaceholder.core.generation.generator import (
    StructureGenerator,
    StructureGeneratorConfigBase,
)
from openplaceholder.core.structure import StructureSet
from openplaceholder.impl.generator.archiver import (
    JSONArchiver,
    JSONArchiverConfig,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JSONGeneratorConfig(StructureGeneratorConfigBase):
    path: str


class JSONGenerator(StructureGenerator):

    _config: JSONGeneratorConfig

    def _setup(self) -> None:
        self._archiver: JSONArchiver = JSONArchiver(JSONArchiverConfig(path=self._config.path))

    def _run(self) -> StructureSet:
        logger.debug("deferring to JSONArchiver for artifact generation")
        return self._archiver.read()

    def _validate_inputs(self) -> None:
        if not Path(self._config.path).exists():
            raise FileNotFoundError("JSON archive does not exist")
