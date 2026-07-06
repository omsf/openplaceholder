import logging
from dataclasses import dataclass
from pathlib import Path

from openplaceholder.core.generation.generator import (
    StructureGenerator,
    StructureGeneratorArtifact,
    StructureGeneratorConfigBase,
)
from openplaceholder.impl.generator.archiver import (
    DirectoryArchiver,
    DirectoryArchiverConfig,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DirectoryGeneratorConfig(StructureGeneratorConfigBase):
    path: str


class DirectoryGenerator(StructureGenerator):

    _config: DirectoryGeneratorConfig

    def _setup(self) -> None:
        self._archiver: DirectoryArchiver = DirectoryArchiver(DirectoryArchiverConfig(path=self._config.path))
        self.validate_inputs()

    def _run(self) -> list[StructureGeneratorArtifact]:
        logger.debug("deferring to DirectoryArchiver for artifact generation")
        return self._archiver.read()

    def _validate_inputs(self) -> None:
        if not Path(self._config.path).exists():
            raise FileNotFoundError("Directory archive does not exist")
