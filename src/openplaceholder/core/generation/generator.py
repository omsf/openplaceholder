from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

from openplaceholder.core.structure.structure import Structure, StructureSet


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


class StructureGenerator(ABC):

    @abstractmethod
    def run(self) -> list[StructureGeneratorArtifact]:
        raise NotImplementedError

    @abstractmethod
    def validate_input(self) -> None:
        raise NotImplementedError
