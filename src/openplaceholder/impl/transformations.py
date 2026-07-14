"""Structure transformations that assemble co-folded complexes for FEP.

Stacked (see ``core.pipeline.Pipeline`` / ``core.runner``): each takes the
previous stage's ``list[Structure]`` and returns a new one, so a user can opt
into only the stages they want.

The substitution transformation gives every complex the *same* protein context
(the canonical one), which makes the preparation and protonation transformations
that follow order-independent -- they operate on every structure identically,
with no positional/"index 0" special-casing. Because all proteins are then
equal, preparing/protonating each one repeats the same work N times; that
duplication is deliberate for now and a later PR can cache it away.

Transformation output is always PDB: protonation adds hydrogens and CONECT
records, so a hydrogenated PDB is the natural output format even if the co-folded
input was MMCIF.
"""

import io
from dataclasses import dataclass

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis import align
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
class MaxVolumeSiteSubstitutionTransformationConfig(TransformationConfigBase):
    pass


class MaxVolumeSiteSubstitutionTransformation(Transformation):
    """Give every complex the same (canonical) protein context.

    Picks the complex whose ligand occupies the largest convex-hull volume as
    the canonical protein, superposes every complex onto it (protein CA fit,
    which rigidly carries each ligand into the canonical frame), and rebuilds
    each structure as the canonical protein + its own now co-framed ligand.
    After this step all structures share one protein, so the downstream
    preparation and protonation stages don't need to track which structure is
    canonical -- they act on every structure identically.
    """

    _config: MaxVolumeSiteSubstitutionTransformationConfig

    def _setup(self) -> None:
        pass

    def _transform(self, structures: list[Structure]) -> list[Structure]:
        if not structures:
            return structures
        volumes = [_ligand_volume(s) for s in structures]
        reference = structures[int(np.argmax(volumes))].to_mda_universe()
        canonical_protein = reference.select_atoms("protein")
        return [self._substitute(s, reference, canonical_protein) for s in structures]

    @staticmethod
    def _substitute(structure: Structure, reference: mda.Universe, canonical_protein: mda.AtomGroup) -> Structure:
        # rigid-fit this complex's protein onto the canonical protein (CA fit),
        # which carries its ligand into the canonical frame, then pair the
        # canonical protein with that now co-framed ligand
        mobile = structure.to_mda_universe()
        align.alignto(mobile, reference, select="protein and name CA")
        return _rebuild(structure, canonical_protein, mobile.select_atoms("not protein"))


@dataclass(frozen=True)
class ProteinPreparationTransformationConfig(TransformationConfigBase):
    pass


class ProteinPreparationTransformation(Transformation):
    """Add missing heavy atoms to each complex's protein.

    PDBFixer fills in missing heavy atoms (and would build missing
    residues/loops for crystal inputs), applied to every structure
    independently. Hydrogens are not added here -- that is
    ``ComplexProtonationTransformation``'s job.
    """

    _config: ProteinPreparationTransformationConfig

    def _setup(self) -> None:
        pass

    def _transform(self, structures: list[Structure]) -> list[Structure]:
        return [self._prepare_protein(s) for s in structures]

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
    """Protonate every complex's ligand and protein at ``ph``.

    Applied to every structure independently. Protein and ligand protonation are
    independent (the ligand method never sees the protein and vice versa), so the
    ligand-protein interface protonation is not guaranteed self-consistent (e.g.
    both partners of an H-bond may end up protonated); reconciling that interface
    is left for a downstream/dedicated step.
    """

    _config: ComplexProtonationTransformationConfig

    def _setup(self) -> None:
        self._ligand_protonator = ProtonateUtilsLigandProtonator()
        self._protein_protonator = ProtonateUtilsProteinProtonator()

    def _transform(self, structures: list[Structure]) -> list[Structure]:
        return [self._protonate_protein(self._protonate_ligand(s)) for s in structures]

    def _protonate_ligand(self, structure: Structure) -> Structure:
        mol = self._ligand_protonator.protonate(structure.to_rdkit_ligand_mol(selection=_LIGAND), self._config.ph)
        ligand = atoms_from_pdb_block(Chem.MolToPDBBlock(mol))
        return _rebuild(structure, structure.protein_atoms(), ligand)

    def _protonate_protein(self, structure: Structure) -> Structure:
        protonated_block = self._protein_protonator.protonate(to_pdb_block(structure.protein_atoms()), self._config.ph)
        return _rebuild(structure, atoms_from_pdb_block(protonated_block), structure.ligand_atoms())
