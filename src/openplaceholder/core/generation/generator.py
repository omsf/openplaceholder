from abc import ABC, abstractmethod
from dataclasses import dataclass

from openplaceholder.core.structure.structure import Structure


@dataclass(frozen=True, eq=True)
class StructureGeneratorArtifact:
    structures: list[Structure]


class StructureGenerator(ABC):

    @abstractmethod
    def run(self) -> list[StructureGeneratorArtifact]:
        raise NotImplementedError

    @abstractmethod
    def validate_input(self) -> None:
        raise NotImplementedError
