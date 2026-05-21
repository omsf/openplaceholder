import base64
import json
from dataclasses import dataclass
from pathlib import Path

from openplaceholder.core.generation.archive import (
    DirectoryArchive,
    DirectoryArchiveConfig,
)
from openplaceholder.core.generation.generator import (
    StructureGenerator,
    StructureGeneratorArtifact,
)
from openplaceholder.core.structure.structure import Structure, StructureFormat


@dataclass(frozen=True, eq=True)
class DirectoryGeneratorConfig:
    directory: str


class DirectoryGenerator(StructureGenerator):

    def __init__(self, config: DirectoryGeneratorConfig):
        self._config = config
        self._archive = DirectoryArchive(DirectoryArchiveConfig(directory=config.directory))
        self.validate_input()

    def run(self) -> list[StructureGeneratorArtifact]:
        return self._archive.read()

    def validate_input(self) -> None:
        if not Path(self._config.directory).exists():
            raise FileNotFoundError("Directory archive does not exist")
