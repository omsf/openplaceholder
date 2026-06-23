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


@dataclass(frozen=True)
class DirectoryGeneratorConfig(StructureGeneratorConfigBase):
    directory: str


class DirectoryGenerator(StructureGenerator):

    _config: DirectoryGeneratorConfig

    def _setup(self) -> None:
        self._archiver: DirectoryArchiver = DirectoryArchiver(DirectoryArchiverConfig(directory=self._config.directory))
        self.validate_inputs()

    def _run(self) -> list[StructureGeneratorArtifact]:
        return self._archiver.read()

    def _validate_inputs(self) -> None:
        if not Path(self._config.directory).exists():
            raise FileNotFoundError("Directory archive does not exist")
