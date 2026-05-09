from dataclasses import dataclass

from openplaceholder.core.generation.generator import (
    StructureGenerator,
    StructureGeneratorArtifact,
)


@dataclass(frozen=True, eq=True)
class DirectoryGeneratorConfig:
    directory: str


@dataclass(frozen=True, eq=True)
class OpenFold3GeneratorConfig:
    sequence: str
    ligands: dict[str, str]
    n_structures: int = 5


class DirectoryGenerator(StructureGenerator):

    def __init__(self, config: DirectoryGeneratorConfig):
        self._config = config

    def run(self) -> list[StructureGeneratorArtifact]:
        raise NotImplementedError

    def validate_input(self) -> None:
        raise NotImplementedError


class OpenFold3Generator(StructureGenerator):

    def __init__(self, config: OpenFold3GeneratorConfig):
        self._config = config

    def run(self) -> list[StructureGeneratorArtifact]:
        raise NotImplementedError

    def validate_input(self) -> None:
        raise NotImplementedError
