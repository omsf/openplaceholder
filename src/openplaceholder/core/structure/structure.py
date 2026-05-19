"""Structure definitions."""

import base64
import hashlib
from dataclasses import dataclass, fields
from enum import StrEnum
from typing import Self


class StructureFormat(StrEnum):
    MMCIF = "MMCIF"
    PDB = "PDB"

    def to_suffix(self) -> str:
        return f".{self.value.lower()}"

    @classmethod
    def from_suffix(cls, suffix: str) -> Self:
        match suffix.lower():
            case ".mmcif" | ".cif":
                return cls.MMCIF
            case ".pdb":
                return cls.PDB
            case _:
                raise ValueError(f"Unsupported structure suffix: '{suffix}'")


@dataclass(frozen=True, eq=True)
class Structure:
    sequence: str
    ligand_smiles: str
    ligand_name: str
    structure_format: str
    structure_data: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "structure_format", StructureFormat(self.structure_format.upper()).value)

    @property
    def structure(self) -> bytes:
        return base64.b64decode(self.structure_data.encode())

    @property
    def key(self) -> str:
        parts = [getattr(self, f.name) for f in fields(self)]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()
