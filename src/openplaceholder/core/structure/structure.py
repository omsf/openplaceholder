"""Structure definitions."""

from dataclasses import dataclass
from enum import StrEnum


class StructureFormat(StrEnum):
    PDB = "PDB"


@dataclass(frozen=True, eq=True)
class Structure:
    structure_ref: str
    sequence: str
    ligand_smiles: str
    ligand_name: str
    structure_format: StructureFormat
