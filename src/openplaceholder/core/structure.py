"""Structure definitions."""

import base64
import contextlib
import hashlib
import io
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import MDAnalysis as mda
from MDAnalysis import Universe
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds
from rdkit.Geometry import Point3D

from openplaceholder.core.mda_pdb import to_pdb_block
from openplaceholder.core.serialization import JSONSerializable, to_shallow_dict


@contextlib.contextmanager
def _quiet_rdkit_warnings() -> Iterator[None]:
    """Silence RDKit's C++ ``rdApp.warning`` logger for the duration of the block.

    ``AssignBondOrdersFromTemplate`` logs "More than one matching pattern found
    - picking one" whenever the template maps onto the perceived skeleton in more
    than one way. Every ligand we handle contains a symmetric group (e.g. the
    2,6-dichlorophenyl of the TYK2 series), so that is essentially always true
    and the message fires once per call -- pure noise. Genuine failures are not
    hidden: they raise, and we re-report them as ``LigandPerceptionError`` with
    RDKit's own message attached.
    """
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.warning")  # type: ignore[attr-defined]
    try:
        yield
    finally:
        RDLogger.EnableLog("rdApp.warning")  # type: ignore[attr-defined]


class UnsupportedFormatError(Exception):
    pass


class LigandPerceptionError(Exception):
    """Raised when a ligand cannot be reconstructed from its 3D pose.

    Distance-based bond perception on a distorted predicted pose can yield a
    connectivity graph that the SMILES template will not map onto; such a pose
    is chemically unusable and should be filtered out rather than crash a run.
    """


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


@dataclass(frozen=True)
class Structure(JSONSerializable):
    sequence: str
    ligand_smiles: str
    ligand_name: str
    structure_format: str
    structure_data: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "structure_format", StructureFormat(self.structure_format.upper()).value)

    def key(self) -> str:
        parts = [getattr(self, f.name) for f in fields(self)]
        return hashlib.sha256("\x00".join(parts).encode()).hexdigest()

    def same_complex(self, other: Self) -> bool:
        raise NotImplementedError

    def decode_structure_data(self) -> bytes:
        return base64.b64decode(self.structure_data.encode("utf-8"))

    def to_mda_universe(self) -> Universe:
        match self.structure_format:
            case StructureFormat.PDB:
                stream = io.StringIO(self.decode_structure_data().decode())
                topology_format = "pdb"
            case StructureFormat.MMCIF:
                stream = io.StringIO(self.decode_structure_data().decode())
                topology_format = "mmcif"
            case _:
                raise UnsupportedFormatError(
                    f"{self.structure_format} is not supported by MDAnalysis ({mda.__version__})."
                )
        return mda.Universe(stream, topology_format=topology_format)

    def to_rdkit_ligand_mol(self, selection: str | None = None) -> Chem.Mol:
        """Build an RDKit Mol for the ligand, with bond orders and
        stereochemistry assigned from ``ligand_smiles`` and 3D coordinates
        taken from the structure.

        Atoms and positions are read directly from MDAnalysis rather than
        via ``AtomGroup.convert_to("RDKIT")``: that path guesses bonds from
        atom *names*, which fails for predicted ligands (generic atom names
        like "C11"/"CL1" aren't recognized as element symbols) and requires
        explicit hydrogens. Connectivity instead comes from RDKit's own
        distance-based ``DetermineConnectivity``, which only needs elements
        and 3D coordinates.

        By default the ligand is the non-protein heavy atoms -- the whole ligand
        in a single-ligand complex, and robust to how the residue was named or
        truncated on write (a PDB ``resName`` is only three characters, so a
        longer ``ligand_name`` cannot survive there; the identity lives in the
        ``ligand_name`` field, not the residue name). Pass ``selection`` to
        override, e.g. ``f"resname {self.ligand_name}"`` to isolate one residue
        from other heteroatoms.
        """
        ligand = self.to_mda_universe().select_atoms(selection or "not protein and not element H")
        if not len(ligand):
            raise LigandPerceptionError(f"no atoms selected for ligand '{self.ligand_name}'")

        editable = Chem.RWMol()
        conformer = Chem.Conformer(len(ligand))
        for i, atom in enumerate(ligand):
            editable.AddAtom(Chem.Atom(atom.element))
            conformer.SetAtomPosition(i, Point3D(*atom.position.astype(float)))
        editable.AddConformer(conformer)

        mol = editable.GetMol()
        try:
            rdDetermineBonds.DetermineConnectivity(mol)

            template = Chem.MolFromSmiles(self.ligand_smiles)
            if template is None:
                raise LigandPerceptionError(f"could not parse ligand_smiles {self.ligand_smiles!r}")
            with _quiet_rdkit_warnings():
                mol = AllChem.AssignBondOrdersFromTemplate(template, mol)  # type: ignore[no-untyped-call]

            # bonds were just (re)assigned from the template, so implicit-H/radical
            # bookkeeping left over from the bond-free starting point is stale;
            # clear it so sanitization fills valences (and hydrogen counts) from
            # the now-correct bond orders.
            editable = Chem.RWMol(mol)
            for mol_atom in editable.GetAtoms():
                mol_atom.SetNoImplicit(False)
                mol_atom.SetNumExplicitHs(0)
                mol_atom.SetNumRadicalElectrons(0)
            mol = editable.GetMol()
            Chem.SanitizeMol(mol)
            # the template can map onto the perceived skeleton more than one way
            # (symmetry, or wrong connectivity); RDKit silently picks one, so
            # verify we got the molecule we asked for rather than trust the pick
            if Chem.MolToSmiles(mol, isomericSmiles=False) != Chem.MolToSmiles(template, isomericSmiles=False):
                raise LigandPerceptionError(f"perceived ligand '{self.ligand_name}' does not match its template")
            Chem.AssignStereochemistryFrom3D(mol)
        except (ValueError, Chem.AtomValenceException, Chem.KekulizeException) as exc:
            raise LigandPerceptionError(f"could not perceive ligand '{self.ligand_name}' from its pose: {exc}") from exc
        return mol

    def protein_atoms(self) -> mda.AtomGroup:
        return self.to_mda_universe().select_atoms("protein")

    def ligand_atoms(self) -> mda.AtomGroup:
        return self.to_mda_universe().select_atoms("not protein")

    def with_atoms(self, atoms: mda.AtomGroup) -> Self:
        """Return a copy of this structure whose data is ``atoms`` serialised as PDB.

        Keeps the metadata (sequence, ligand) and normalises the format to PDB --
        the write-side mirror of :meth:`to_mda_universe`.
        """
        block = to_pdb_block(atoms)
        return replace(
            self,
            structure_format=StructureFormat.PDB,
            structure_data=base64.b64encode(block.encode()).decode(),
        )

    def to_dict(self) -> dict[Any, Any]:
        return to_shallow_dict(self)

    @classmethod
    def from_dict(cls, data: dict[Any, Any]) -> Self:
        data.pop("__oph_custom__", None)
        return cls(**data)


@dataclass(frozen=True)
class StructureSet(JSONSerializable):
    structures: list[Structure]

    def __post_init__(self) -> None:
        object.__setattr__(self, "structures", sorted(self.structures, key=lambda s: s.key()))

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

    def to_dict(self) -> dict[Any, Any]:
        return to_shallow_dict(self)

    @classmethod
    def from_dict(cls, data: dict[Any, Any]) -> Self:
        data.pop("__oph_custom__", None)
        return cls(**data)

    def __len__(self) -> int:
        return len(self.structures)
