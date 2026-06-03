"""Structure definitions."""

import base64
import hashlib
import json
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import Self

from MDAnalysis import Universe


class StructureFormat(StrEnum):
    MMCIF = "MMCIF"
    PDB = "PDB"

    @staticmethod
    def supported_formats() -> set[str]:
        return {sf.to_suffix() for sf in StructureFormat}

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
        return hashlib.sha256("\x00".join(parts).encode()).hexdigest()

    def same_complex(self, other: Self) -> bool:
        raise NotImplementedError

    def to_mda(self) -> "Universe":
        raise NotImplementedError


@dataclass(frozen=True, eq=True)
class StructureSet:
    structures: list[Structure]

    def __post_init__(self) -> None:
        object.__setattr__(self, "structures", sorted(self.structures, key=lambda s: s.key))

    @classmethod
    def from_structures(cls, structures: list[Structure]) -> Self:
        return cls(structures=[*{*structures}])

    @classmethod
    def from_file(cls, file_path: str | Path) -> Self:
        file_path = Path(file_path)
        content = json.loads(file_path.read_text())
        structures = [Structure(**structure) for structure in content["structures"]]
        artifact_data = content | {"structures": structures}
        return cls(**artifact_data)

    def write(self, file_path: str | Path) -> None:
        file_path = Path(file_path)
        with open(file_path, "w") as f:
            json.dump(asdict(self), f)

    def __len__(self) -> int:
        return len(self.structures)
