"""Structure definitions."""

import base64
import contextlib
import hashlib
import io
import json
from dataclasses import asdict, dataclass, fields, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import MDAnalysis as mda
import numpy as np
from MDAnalysis import Universe
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds
from rdkit.Geometry import Point3D

from openplaceholder.core.serialization import JSONSerializable, to_shallow_dict
from openplaceholder.core.utils import _quiet_rdkit_warnings


def atoms_to_pdb_string(atoms: mda.AtomGroup) -> str:
    """Serialise an AtomGroup to a PDB string."""
    buffer = io.StringIO()
    with mda.Writer(buffer, format="PDB", n_atoms=len(atoms)) as writer:
        writer.write(atoms)
        # the MMCIF parser's default altLoc is a NUL byte that corrupts the
        # fixed-width PDB columns MDAnalysis writes it into and breaks the PDB
        # parsers downstream; scrub it on every round-trip
        return buffer.getvalue().replace("\x00", " ")


def atoms_from_pdb_string(block: str) -> mda.AtomGroup:
    """Parse a PDB string into an MDAnalysis AtomGroup."""
    return mda.Universe(io.StringIO(block), topology_format="PDB").atoms


def _attach_hydrogens(mol: Chem.Mol, heavy: mda.AtomGroup, hydrogens: mda.AtomGroup) -> Chem.Mol:
    """Add `hydrogens` to a perceived heavy-atom `mol`, keeping their real coordinates.

    Each hydrogen is bonded to its nearest heavy atom. Doing it this
    way keeps hydrogens out of ``DetermineConnectivity`` and the
    template match, where they perturb the mapping and produce
    spurious valence errors.
    """
    editable = Chem.RWMol(mol)
    positions = [list(mol.GetConformer().GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    for hydrogen in hydrogens:
        parent_index = int(np.argmin(np.linalg.norm(heavy.positions - hydrogen.position, axis=1)))
        editable.AddBond(parent_index, editable.AddAtom(Chem.Atom(1)), Chem.BondType.SINGLE)
        positions.append(list(hydrogen.position.astype(np.float64)))

    # the hydrogens are explicit now, decide each heavy atom's
    # hydrogen count
    for mol_atom in editable.GetAtoms():
        if mol_atom.GetAtomicNum() > 1:
            mol_atom.SetNoImplicit(True)
            mol_atom.SetNumExplicitHs(0)

    mol = editable.GetMol()
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for i, position in enumerate(positions):
        conformer.SetAtomPosition(i, Point3D(*position))
    mol.RemoveAllConformers()
    mol.AddConformer(conformer)
    Chem.SanitizeMol(mol)
    return mol


class UnsupportedFormatError(Exception):
    pass


class LigandPerceptionError(Exception):
    """Raised when a ligand cannot be reconstructed from its 3D pose.

    Distance-based bond perception on a distorted predicted pose can yield a
    connectivity graph that the SMILES template will not map onto; such a pose
    is chemically unusable.
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

        By default the ligand is everything that isn't protein -- the whole
        ligand in a single-ligand complex, and robust to how the residue was
        named or truncated on write (a PDB ``resName`` is only three characters,
        so a longer ``ligand_name`` cannot survive there; the identity lives in
        the ``ligand_name`` field, not the residue name). Any hydrogens it has
        are kept: an unprotonated pose simply has none, while a protonated one
        is returned as stored rather than silently stripped. Pass ``selection``
        to override, e.g. ``f"resname {self.ligand_name}"`` to isolate one
        residue from other heteroatoms.
        """

        ligand = self.to_mda_universe().select_atoms(selection or "not protein")
        if not len(ligand):
            raise LigandPerceptionError(f"no atoms selected for ligand '{self.ligand_name}'")

        # perceive from the heavy atoms alone; any explicit hydrogens the caller
        # selected are re-attached afterwards (see _attach_hydrogens)
        heavy = ligand.select_atoms("not element H")
        hydrogens = ligand.select_atoms("element H")

        editable = Chem.RWMol()
        conformer = Chem.Conformer(len(heavy))
        for i, atom in enumerate(heavy):
            editable.AddAtom(Chem.Atom(atom.element))
            conformer.SetAtomPosition(i, Point3D(*atom.position.astype(float)))
        editable.AddConformer(conformer)

        mol = editable.GetMol()
        try:
            rdDetermineBonds.DetermineConnectivity(mol)

            template = Chem.MolFromSmiles(self.ligand_smiles)
            if template is None:
                raise LigandPerceptionError(f"could not parse ligand_smiles {self.ligand_smiles!r}")

            # most systems we handle will have a symmetric group, producing excessive
            # rdkit warnings about multiple matching patterns.
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
            # only if the caller selected them: the heavy-atom default has none
            # and keeps its implicit hydrogens, while a selection that spans a
            # protonated ligand gets those hydrogens back rather than silently
            # dropping the protonation it asked for
            if len(hydrogens):
                mol = _attach_hydrogens(mol, heavy, hydrogens)
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
        block = atoms_to_pdb_string(atoms)
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
