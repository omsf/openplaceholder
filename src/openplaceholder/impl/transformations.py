"""Structure transformations that assemble co-folded complexes for FEP.

These transformations are meant to be *stacked* (see ``core.pipeline.Pipeline``
and ``core.runner``): each takes the previous stage's ``list[Structure]`` and
returns a new one. Splitting the work into separate transformations lets a user
opt into only the stages they want -- e.g. prepare without protonating, or
protonate a set that was selected elsewhere.

Ordering contract: **index 0 is the canonical protein context.** The selection
transformation establishes it (the largest-ligand-volume complex); the
preparation and protonation transformations act on that one protein and leave
the rest as ligand carriers. Each transformation reads ``structures[0]`` on a
single explicit line; every helper below operates on one ``Structure`` at a
time and never on list position. If selection is omitted from a pipeline, the
contract still holds -- ``structures[0]`` is simply whatever the upstream stage
emitted.

Transformation output is always PDB: protonation adds hydrogens and CONECT
records, so a hydrogenated PDB is the natural, deliberate output format even if
the co-folded input was MMCIF.
"""

import base64
import io
from dataclasses import dataclass

import MDAnalysis as mda
import numpy as np
from openmm.app import PDBFile
from pdbfixer import PDBFixer
from rdkit import Chem
from scipy.spatial import ConvexHull

from openplaceholder.core.assembly.transformation import (
    Transformation,
    TransformationConfigBase,
)
from openplaceholder.core.structure import Structure, StructureFormat
from openplaceholder.impl.protonation import (
    ProlifInterfaceReconciler,
    ProtonateUtilsLigandProtonator,
    ProtonateUtilsProteinProtonator,
)

# heavy-atom ligand of a complex, robust to truncated residue names
_LIGAND = "not protein and not element H"


def _protein_atoms(structure: Structure) -> mda.AtomGroup:
    return structure.to_mda_universe().select_atoms("protein")


def _ligand_atoms(structure: Structure) -> mda.AtomGroup:
    return structure.to_mda_universe().select_atoms("not protein")


def _ligand_volume(structure: Structure) -> float:
    return float(ConvexHull(_ligand_atoms(structure).positions).volume)


def _to_pdb_block(atoms: mda.AtomGroup) -> str:
    """Serialise an AtomGroup to a PDB string in memory (no disk I/O)."""
    buffer = io.StringIO()
    with mda.Writer(buffer, format="PDB", n_atoms=len(atoms)) as writer:
        writer.write(atoms)
        # the MMCIF parser's default altLoc is a NUL byte that corrupts the
        # fixed-width PDB columns MDAnalysis writes it into and breaks the PDB
        # parsers downstream; scrub it on every round-trip
        return buffer.getvalue().replace("\x00", " ")


def _atoms_from_pdb_block(block: str) -> mda.AtomGroup:
    """Parse a PDB string back into an AtomGroup in memory (no disk I/O)."""
    return mda.Universe(io.StringIO(block), topology_format="PDB").atoms


def _assemble(template: Structure, protein: mda.AtomGroup, ligand: mda.AtomGroup) -> Structure:
    """Build a PDB ``Structure`` from protein + ligand atoms, keeping metadata."""
    block = _to_pdb_block(mda.Merge(protein, ligand).atoms)
    return Structure(
        sequence=template.sequence,
        ligand_smiles=template.ligand_smiles,
        ligand_name=template.ligand_name,
        # PDB is the deliberate output format for a hydrogenated complex
        structure_format=StructureFormat.PDB,
        structure_data=base64.b64encode(block.encode()).decode(),
    )


@dataclass(frozen=True)
class MaxVolumeSiteSelectionTransformationConfig(TransformationConfigBase):
    pass


class MaxVolumeSiteSelectionTransformation(Transformation):
    """Reorder complexes so the canonical protein context is first.

    Picks the complex whose ligand occupies the largest convex-hull volume and
    returns the list with that complex at index 0 (establishing the ordering
    contract the preparation/protonation stages rely on). Pure reordering: no
    structure is prepared, protonated, or otherwise altered here.
    """

    _config: MaxVolumeSiteSelectionTransformationConfig

    def _setup(self) -> None:
        pass

    def _transform(self, structures: list[Structure]) -> list[Structure]:
        if not structures:
            return structures
        volumes = [_ligand_volume(s) for s in structures]
        chosen = int(np.argmax(volumes))
        return [structures[chosen], *(s for i, s in enumerate(structures) if i != chosen)]


@dataclass(frozen=True)
class ProteinPreparationTransformationConfig(TransformationConfigBase):
    pass


class ProteinPreparationTransformation(Transformation):
    """Add missing heavy atoms to the canonical protein.

    PDBFixer fills in missing heavy atoms (and would build missing
    residues/loops for crystal inputs) on the protein of ``structures[0]``, the
    canonical protein context. Hydrogens are not added here -- that is
    ``ComplexProtonationTransformation``'s job. Other structures are untouched.
    """

    _config: ProteinPreparationTransformationConfig

    def _setup(self) -> None:
        pass

    def _transform(self, structures: list[Structure]) -> list[Structure]:
        if not structures:
            return structures
        canonical = structures[0]  # ordering contract: index 0 is the canonical protein context
        return [self._prepare_protein(canonical), *structures[1:]]

    def _prepare_protein(self, structure: Structure) -> Structure:
        fixer = PDBFixer(pdbfile=io.StringIO(_to_pdb_block(_protein_atoms(structure))))
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()  # heavy atoms only; hydrogens come from ComplexProtonationTransformation

        sink = io.StringIO()
        PDBFile.writeFile(fixer.topology, fixer.positions, sink)
        fixed = _atoms_from_pdb_block(sink.getvalue())
        return _assemble(structure, fixed, _ligand_atoms(structure))


@dataclass(frozen=True)
class ComplexProtonationTransformationConfig(TransformationConfigBase):
    # protonation pH for both protein and ligand
    ph: float = 7.0


class ComplexProtonationTransformation(Transformation):
    """Protonate every ligand and the canonical protein, then reconcile them.

    Every ligand is protonated at ``ph`` (position-independent); the canonical
    protein (``structures[0]``) is protonated at ``ph`` and its interface with
    its ligand is reconciled.

    Protein and ligand protonation are independent: the ligand method never sees
    the protein and vice versa. Where a ligand-protein hydrogen bond spans the
    two, neither side accounts for the other, so both partners can end up
    protonated (a donor-donor clash) or both left bare (an acceptor-acceptor
    miss). The reconcile step removes the protein-side hydrogen of any
    donor-donor clash and warns about acceptor-acceptor misses (which need a
    proton invented, so they are left for review). Backbone-amide clashes and
    all acceptor-acceptor misses are pocket-pKa problems no local rule fixes.
    """

    _config: ComplexProtonationTransformationConfig

    def _setup(self) -> None:
        self._ligand_protonator = ProtonateUtilsLigandProtonator()
        self._protein_protonator = ProtonateUtilsProteinProtonator()
        self._reconciler = ProlifInterfaceReconciler()

    def _transform(self, structures: list[Structure]) -> list[Structure]:
        if not structures:
            return structures
        # protonate every ligand first, while proteins still carry standard
        # residue names, then protonate + reconcile the canonical protein
        protonated = [self._protonate_ligand(s) for s in structures]
        canonical = protonated[0]  # ordering contract: index 0 is the canonical protein context
        return [self._reconcile(self._protonate_protein(canonical)), *protonated[1:]]

    def _protonate_ligand(self, structure: Structure) -> Structure:
        mol = self._ligand_protonator.protonate(structure.to_rdkit_ligand_mol(selection=_LIGAND), self._config.ph)
        ligand = _atoms_from_pdb_block(Chem.MolToPDBBlock(mol))
        return _assemble(structure, _protein_atoms(structure), ligand)

    def _protonate_protein(self, structure: Structure) -> Structure:
        protonated_block = self._protein_protonator.protonate(_to_pdb_block(_protein_atoms(structure)), self._config.ph)
        return _assemble(structure, _atoms_from_pdb_block(protonated_block), _ligand_atoms(structure))

    def _reconcile(self, structure: Structure) -> Structure:
        # the ligand heavy-atom coordinates are unchanged by protonation, so
        # re-protonating them deterministically reproduces the ligand mol (with
        # explicit hydrogens + conformer) the reconciler needs
        ligand_mol = self._ligand_protonator.protonate(
            structure.to_rdkit_ligand_mol(selection=_LIGAND), self._config.ph
        )
        fixed_protein = self._reconciler.reconcile(_protein_atoms(structure), ligand_mol)
        return _assemble(structure, fixed_protein, _ligand_atoms(structure))
