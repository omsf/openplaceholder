"""Structure definitions."""

import base64
import io
import logging
from enum import StrEnum
from functools import cache
from typing import Any, Iterator, Self, cast

import MDAnalysis as mda
import numpy as np
from gufe.tokenization import GufeTokenizable
from MDAnalysis import Universe
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds
from rdkit.Geometry import Point3D

from openplaceholder.core.utils import _quiet_rdkit_warnings

logger = logging.getLogger(__name__)


def atoms_to_pdb_string(atoms: mda.AtomGroup) -> str:
    """Serialise an AtomGroup to a PDB string.

    Parameters
    ----------
    atoms
        MDAnalysis AtomGroup to write to the PDB format.

    Returns
    -------
    A string with the encoded PDB.

    """
    with io.StringIO() as buffer:
        with mda.Writer(buffer, format="PDB", n_atoms=len(atoms)) as writer:
            writer.write(atoms)
            # grab value before the writer closes the buffer
            s = buffer.getvalue()
        return s


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
    for mol_atom in editable.GetAtoms():  # type: ignore
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
    PDB = "PDB"

    @staticmethod
    def supported_formats() -> set[str]:
        return {sf.to_suffix() for sf in StructureFormat}

    def to_suffix(self) -> str:
        return f".{self.value.lower()}"

    @classmethod
    def from_suffix(cls, suffix: str) -> "StructureFormat":
        """Determine and return the StructureFormat from a suffix.

        Note that this assumes the creator of the file paired the
        correct suffix and format when writing the file. No validation
        is performed.

        Parameters
        ----------
        suffix
            The suffix string for StructureFormat determination.

        Returns
        -------
        StructureFormat

        Raises
        ------
        UnsupportedFormatError
        """
        match suffix.lower():
            case ".pdb":
                return cls.PDB
            case _:
                raise UnsupportedFormatError(f"Unsupported structure suffix: '{suffix}'")


class Structure(GufeTokenizable):

    def __init__(
        self, sequence: str, ligand_smiles: str, ligand_name: str, structure_format: str, structure_data: str
    ) -> None:
        self.sequence = sequence
        self.ligand_smiles = ligand_smiles
        self.ligand_name = ligand_name
        self.structure_format = StructureFormat(structure_format.upper()).value
        self.structure_data = structure_data
        super().__init__()

    @classmethod
    def _defaults(cls) -> dict[Any, Any]:
        return {
            "sequence": "",
            "ligand_smiles": "",
            "ligand_name": "",
            "structure_format": StructureFormat.PDB,
            "structure_data": "",
        }

    def _to_dict(self) -> dict[Any, Any]:
        return {
            "sequence": self.sequence,
            "ligand_smiles": self.ligand_smiles,
            "ligand_name": self.ligand_name,
            "structure_format": self.structure_format,
            "structure_data": self.structure_data,
        }

    @classmethod
    def _from_dict(cls, dct: dict[Any, Any]) -> Self:
        return cls(**dct)

    def same_complex(self, other: Self) -> bool:
        _ = other
        raise NotImplementedError

    def decode_structure_data(self) -> bytes:
        return base64.b64decode(self.structure_data.encode("utf-8"))

    @cache
    def to_mda_universe(self) -> Universe:
        """Build an MDAnalysis Universe from the structure data.

        MDAnalysis usually guesses file formats from the suffix of a
        file. Since a ``Structure`` instance holds raw bytes of a
        structure, the ``Structure.structure_format`` is used in leiu
        of a file suffix.

        This is a cached function.

        Returns
        -------
        Universe
            An MDAnalysis constructed constructed from the
            ``structure_data`` and ``structure_format``.

        Raises
        ------
        UnsupportedFormat
        """
        match self.structure_format:
            case StructureFormat.PDB:
                stream = io.StringIO(self.decode_structure_data().decode())
                topology_format = "pdb"
            case _:
                raise UnsupportedFormatError(
                    f"{self.structure_format} is not supported by MDAnalysis ({mda.__version__})."
                )
        return mda.Universe(stream, topology_format=topology_format)

    @cache
    def to_rdkit_ligand_mol(self, selection: str | None = None) -> Chem.Mol:
        """Build an RDKit Mol for the ligand.

        Bond orders and stereochemistry assigned from
        ``ligand_smiles`` and 3D coordinates taken from the structure.
        Atoms and positions are read directly from MDAnalysis rather
        than via ``AtomGroup.convert_to("RDKIT")``: that path guesses
        bonds from atom *names*, which fails for predicted ligands
        (generic atom names like "C11"/"CL1" aren't recognized as
        element symbols) and requires explicit hydrogens. Connectivity
        instead comes from RDKit's own distance-based
        ``DetermineConnectivity``, which only needs elements and 3D
        coordinates. By default (optionally overrided by
        ``selection``) the ligand is everything that isn't
        protein. Any hydrogens it has are kept.

        This is a cached function.

        Parameters
        ----------
        selection
            An optional string for atom selection.

        Returns
        -------
        Chem.Mol

        Raises
        ------
        LigandPerceptionError
            The ligand was either not recoverable from the MDAnalysis
            Universe or RDKit was unable to fully process the ligand.

        """

        ligand = self.to_mda_universe().select_atoms(selection or "not protein")
        if not len(ligand):
            raise LigandPerceptionError(f"no atoms selected for ligand '{self.ligand_name}'")

        # perceive from the heavy atoms alone; any explicit hydrogens the caller
        # selected are re-attached afterwards (see _attach_hydrogens)
        heavy = ligand.select_atoms("not element H")
        hydrogens = ligand.select_atoms("element H")

        logger.debug("Found %d atoms for ligand %s", len(ligand), self.ligand_name)

        editable = Chem.RWMol()
        conformer = Chem.Conformer(len(heavy))
        for i, atom in enumerate(heavy):
            editable.AddAtom(Chem.Atom(atom.element))
            conformer.SetAtomPosition(i, Point3D(*atom.position.astype(float)))
        editable.AddConformer(conformer)

        mol = editable.GetMol()
        try:
            rdDetermineBonds.DetermineConnectivity(mol)

            template = cast(Chem.Mol | None, Chem.MolFromSmiles(self.ligand_smiles))
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
            for mol_atom in editable.GetAtoms():  # type: ignore
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

    @cache
    def protein_atoms(self) -> mda.AtomGroup:
        return self.to_mda_universe().select_atoms("protein")

    @cache
    def ligand_atoms(self) -> mda.AtomGroup:
        return self.to_mda_universe().select_atoms("not protein")

    def with_atoms(self, atoms: mda.AtomGroup) -> Self:
        """Return a copy of this structure whose data is ``atoms`` serialised as PDB.

        Keeps the metadata (sequence, ligand) and normalises the format to PDB --
        the write-side mirror of :meth:`to_mda_universe`.
        """
        block = atoms_to_pdb_string(atoms)
        return self.copy_with_replacements(
            structure_format=StructureFormat.PDB,
            structure_data=base64.b64encode(block.encode()).decode(),
        )


class StructureSet(GufeTokenizable):
    """A list of Structure instances with convenience methods for serialization."""

    def __init__(self, structures: list[Structure]) -> None:
        self.structures = list(sorted(structures, key=lambda s: s.key))

    def _to_dict(self) -> dict[Any, Any]:
        return {"structures": self.structures}

    @classmethod
    def _from_dict(cls, dct: dict[Any, Any]) -> Self:
        return cls(**dct)

    @classmethod
    def _defaults(cls) -> dict[Any, Any]:
        return {"structures": tuple()}

    def __len__(self) -> int:
        return len(self.structures)

    def __iter__(self) -> Iterator[Structure]:
        yield from self.structures

    def __getitem__(self, key: int) -> Structure:
        return self.structures[key]
