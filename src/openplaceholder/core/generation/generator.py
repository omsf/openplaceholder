from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

from openplaceholder.core.structure.structure import Structure, StructureSet


@dataclass(frozen=True, eq=True)
class StructureGeneratorArtifact(StructureSet):

    sequence: str
    ligand_smiles: str

    @classmethod
    def from_structures(cls, structures: list[Structure]) -> Self:
        structures = StructureSet.from_structures(structures).structures

        sequences = set()
        ligands_smiles = set()
        for structure in structures:
            sequences.add(structure.sequence)
            ligands_smiles.add(structure.ligand_smiles)

        if len(sequences) > 1 or len(ligands_smiles) > 1:
            raise ValueError("Structures do not represent the same complex")

        sequence = sequences.pop()
        ligand_smiles = ligands_smiles.pop()

        return cls(structures=structures, sequence=sequence, ligand_smiles=ligand_smiles)


class StructureGenerator(ABC):

    @abstractmethod
    def run(self) -> list[StructureGeneratorArtifact]:
        raise NotImplementedError

    @abstractmethod
    def validate_input(self) -> None:
        raise NotImplementedError
