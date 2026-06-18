from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

from openplaceholder.core.configuration import ConfigBase
from openplaceholder.core.interface import Configurable, Module
from openplaceholder.core.structure import Structure, StructureSet


@dataclass(frozen=True, eq=True)
class StructureGeneratorArtifact(StructureSet):

    sequence: str
    ligand_smiles: str
    ligand_name: str

    @classmethod
    def from_structures(cls, structures: list[Structure]) -> Self:
        structures = StructureSet.from_structures(structures).structures

        sequences = set()
        ligands_smiles = set()
        ligands_name = set()
        for structure in structures:
            sequences.add(structure.sequence)
            ligands_smiles.add(structure.ligand_smiles)
            ligands_name.add(structure.ligand_name)

        if len(sequences) > 1 or len(ligands_smiles) > 1 or len(ligands_name) > 1:
            raise ValueError("Structures do not represent the same complex")

        sequence = sequences.pop()
        ligand_smiles = ligands_smiles.pop()
        ligand_name = ligands_name.pop()

        return cls(structures=structures, sequence=sequence, ligand_smiles=ligand_smiles, ligand_name=ligand_name)


class ArtifactArchiver(ABC):

    @abstractmethod
    def _write(self, artifacts: list["StructureGeneratorArtifact"]) -> None:
        raise NotImplementedError

    @abstractmethod
    def _read(self) -> list["StructureGeneratorArtifact"]:
        raise NotImplementedError

    @abstractmethod
    def _archive_exists(self) -> bool:
        raise NotImplementedError

    def write(self, artifacts: list["StructureGeneratorArtifact"]) -> None:
        return self._write(artifacts)

    def read(self) -> list["StructureGeneratorArtifact"]:
        return self._read()

    def archive_exists(self) -> bool:
        return self._archive_exists()


class StructureGeneratorConfigBase(ConfigBase):
    archiver = None


class StructureGenerator(Module, ABC):

    _archiver: ArtifactArchiver | None = None

    def __init__(self, *args, **kwargs) -> None:  # type: ignore
        super().__init__(*args, **kwargs)
        self.validate_inputs()

    @abstractmethod
    def _run(self) -> list[StructureGeneratorArtifact]:
        raise NotImplementedError

    @abstractmethod
    def _validate_inputs(self) -> None:
        raise NotImplementedError

    def run(self) -> list[StructureGeneratorArtifact]:

        if self._archiver and self._archiver.archive_exists():
            return self._archiver.read()

        return self._run()

    def validate_inputs(self) -> None:

        if archiver := getattr(self, "_archiver", None):
            if not isinstance(archiver, ArtifactArchiver):
                raise ValueError("archiver attribute for a generator must be an Archiver instance")
        else:
            self._archiver = None

        self._validate_inputs()
