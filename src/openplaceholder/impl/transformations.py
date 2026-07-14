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
from openplaceholder.core.mda_pdb import atoms_from_pdb_block, to_pdb_block
from openplaceholder.core.structure import Structure
from openplaceholder.impl.protonation import (
    ProtonateUtilsLigandProtonator,
    ProtonateUtilsProteinProtonator,
)

# heavy-atom ligand of a complex, robust to truncated residue names
_LIGAND = "not protein and not element H"


def _ligand_volume(structure: Structure) -> float:
    return float(ConvexHull(structure.ligand_atoms().positions).volume)


def _rebuild(structure: Structure, protein: mda.AtomGroup, ligand: mda.AtomGroup) -> Structure:
    """Reassemble a complex from its protein + ligand atoms, keeping metadata."""
    return structure.with_atoms(mda.Merge(protein, ligand).atoms)


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
        fixer = PDBFixer(pdbfile=io.StringIO(to_pdb_block(structure.protein_atoms())))
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()  # heavy atoms only; hydrogens come from ComplexProtonationTransformation

        sink = io.StringIO()
        PDBFile.writeFile(fixer.topology, fixer.positions, sink)
        return _rebuild(structure, atoms_from_pdb_block(sink.getvalue()), structure.ligand_atoms())


@dataclass(frozen=True)
class ComplexProtonationTransformationConfig(TransformationConfigBase):
    # protonation pH for both protein and ligand
    ph: float = 7.0


class ComplexProtonationTransformation(Transformation):
    """Protonate every ligand and the canonical protein.

    Every ligand is protonated at ``ph`` (position-independent); the canonical
    protein (``structures[0]``) is protonated at ``ph``.

    Protein and ligand protonation are independent: the ligand method never sees
    the protein and vice versa. Where a ligand-protein hydrogen bond spans the
    two, neither side accounts for the other, so the interface protonation is not
    guaranteed self-consistent (e.g. both partners of an H-bond may end up
    protonated). This is left for a downstream/dedicated step to address.
    """

    _config: ComplexProtonationTransformationConfig

    def _setup(self) -> None:
        self._ligand_protonator = ProtonateUtilsLigandProtonator()
        self._protein_protonator = ProtonateUtilsProteinProtonator()

    def _transform(self, structures: list[Structure]) -> list[Structure]:
        if not structures:
            return structures
        # protonate every ligand first, while proteins still carry standard
        # residue names, then protonate the canonical protein
        protonated = [self._protonate_ligand(s) for s in structures]
        canonical = protonated[0]  # ordering contract: index 0 is the canonical protein context
        return [self._protonate_protein(canonical), *protonated[1:]]

    def _protonate_ligand(self, structure: Structure) -> Structure:
        mol = self._ligand_protonator.protonate(structure.to_rdkit_ligand_mol(selection=_LIGAND), self._config.ph)
        ligand = atoms_from_pdb_block(Chem.MolToPDBBlock(mol))
        return _rebuild(structure, structure.protein_atoms(), ligand)

    def _protonate_protein(self, structure: Structure) -> Structure:
        protonated_block = self._protein_protonator.protonate(to_pdb_block(structure.protein_atoms()), self._config.ph)
        return _rebuild(structure, atoms_from_pdb_block(protonated_block), structure.ligand_atoms())
