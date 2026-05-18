from dataclasses import dataclass

from openplaceholder.core.generation.generator import (
    StructureGenerator,
    StructureGeneratorArtifact,
)


@dataclass(frozen=True, eq=True)
class DirectoryGeneratorConfig:
    directory: str


class DirectoryGenerator(StructureGenerator):

    def __init__(self, config: DirectoryGeneratorConfig):
        self._config = config

    def run(self) -> list[StructureGeneratorArtifact]:
        raise NotImplementedError

    def validate_input(self) -> None:
        raise NotImplementedError
