try:
    import openfold3
except:
    raise ImportError("Failed to import openfold3. Is it installed?")

from dataclasses import dataclass

from openplaceholder.core.generation.generator import (
    StructureGenerator,
    StructureGeneratorArtifact,
)


@dataclass(frozen=True, eq=True)
class OpenFold3GeneratorConfig:
    sequence: str
    ligands: dict[str, str]
    n_structures: int = 5
    seeds: list[int]
    diffusion_samples: int


class OpenFold3Generator(StructureGenerator):

    def __init__(self, config: OpenFold3GeneratorConfig):
        self._config = config

    def run(self) -> list[StructureGeneratorArtifact]:
        raise NotImplementedError

    def validate_input(self) -> None:
        raise NotImplementedError
