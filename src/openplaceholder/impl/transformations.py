"""Structure transformations that modify co-folded complexes for FEP."""

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
from openplaceholder.core.structure import (
    Structure,
    atoms_to_pdb_string,
)
from openplaceholder.vendor.protonate_utils import (
    protonate_molecule,
    protonate_structure,
)

# TODO: this should be moved to a higher level
# stable PDB residue name for the protonated ligand -- RDKit's MolToPDBBlock
# stamps "UNL" otherwise, and a PDB resName is only three characters (so a
# longer ligand_name cannot be preserved here)
_LIGAND_RESNAME = "LIG"

# hydride's compiled relaxation step (geometry-optimising the placed hydrogens)
# is incompatible with the numpy 2.x stack here (an int32/long buffer mismatch);
# hydrogen *placement* -- the pH-correct states -- is unaffected, so relaxation
# is left off.
_RELAX_PROTEIN_HYDROGENS = False


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
        fixer = PDBFixer(pdbfile=io.StringIO(atoms_to_pdb_string(structure.protein_atoms())))
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()  # heavy atoms only; hydrogens come from ComplexProtonationTransformation

        with io.StringIO() as buffer:
            PDBFile.writeFile(fixer.topology, fixer.positions, buffer)
            u_protonated = mda.Universe(buffer, topology_format="PDB")

        return _rebuild(structure, u_protonated.atoms, structure.ligand_atoms())


@dataclass(frozen=True)
class ComplexProtonationTransformationConfig(TransformationConfigBase):
    # protonation pH for both protein and ligand
    ph: float = 7.0


class ComplexProtonationTransformation(Transformation):
    """Protonate every complex's ligand and protein at ``ph``.

    Applied to every structure independently. Ligand protonation uses
    Dimorphite-DL microstates and protein protonation uses hydride (both vendored
    from PatWalters/protonate_utils). Protein and ligand protonation are
    independent (the ligand method never sees the protein and vice versa), so the
    ligand-protein interface protonation is not guaranteed self-consistent (e.g.
    both partners of an H-bond may end up protonated); reconciling that interface
    is left for a downstream/dedicated step.
    """

    _config: ComplexProtonationTransformationConfig

    def _setup(self) -> None:
        pass

    def _transform(self, structures: list[Structure]) -> list[Structure]:
        return [self._protonate_protein(self._protonate_ligand(s)) for s in structures]

    def _protonate_ligand(self, structure: Structure) -> Structure:
        mol: Chem.Mol = protonate_molecule(structure.to_rdkit_ligand_mol(), self._config.ph)  # type: ignore[no-untyped-call]

        for atom in mol.GetAtoms():
            info = Chem.AtomPDBResidueInfo()
            info.SetResidueName(_LIGAND_RESNAME)
            info.SetIsHeteroAtom(True)
            atom.SetMonomerInfo(info)

        with io.StringIO(Chem.MolToPDBBlock(mol)) as buffer:
            ligand = mda.Universe(buffer, topology_format="PDB").atoms

        return _rebuild(structure, structure.protein_atoms(), ligand)

    def _protonate_protein(self, structure: Structure) -> Structure:
        import biotite.structure.io.pdb as pdb_io

        source = pdb_io.PDBFile.read(io.StringIO(atoms_to_pdb_string(structure.protein_atoms())))
        protonated = protonate_structure(  # type: ignore[no-untyped-call]
            source.get_structure(model=1), ph=self._config.ph, relax=_RELAX_PROTEIN_HYDROGENS
        )

        out_file = pdb_io.PDBFile()
        out_file.set_structure(protonated)
        sink = io.StringIO()
        out_file.write(sink)
        reconstructed = mda.Universe(sink, topology_format="PDB")
        return _rebuild(structure, reconstructed.atoms, structure.ligand_atoms())
