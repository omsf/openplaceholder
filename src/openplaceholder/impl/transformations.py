import base64
import tempfile
from dataclasses import dataclass
from pathlib import Path

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
    ProtonateUtilsLigandProtonator,
    ProtonateUtilsProteinProtonator,
)

# heavy-atom ligand of a complex, robust to truncated residue names
_LIGAND = "not protein and not element H"


@dataclass(frozen=True)
class MaxVolumeSiteTransformationConfig(TransformationConfigBase):
    # protonation pH for both protein and ligand
    ph: float = 7.0


class MaxVolumeSiteTransformation(Transformation):
    """Assemble co-folded complexes into a list of ligands + one protein context.

    Three steps run in sequence:

      1. select     -- choose the single canonical protein as the one whose
                       ligand occupies the largest convex-hull volume.
      2. prepare    -- PDBFixer adds missing heavy atoms to that protein (and
                       would build missing residues/loops for crystal inputs).
      3. protonate  -- add hydrogens at ``ph`` to that protein and to every
                       ligand, using the configured protonation methods.

    Selecting first means only the chosen protein is prepared/protonated.
    Ligands stay in their own frames (no coordinate transplant); the chosen
    protein context is returned first.

    Protein and ligand protonation are independent: the ligand method never
    sees the protein and vice versa. Where a ligand-protein hydrogen bond
    spans the two, neither side accounts for the other, so both partners can
    end up protonated (a donor-donor clash) -- callers needing bonded-network
    consistency should reconcile the interface afterwards.
    """

    _config: MaxVolumeSiteTransformationConfig

    def _setup(self) -> None:
        self._ligand_protonator = ProtonateUtilsLigandProtonator()
        self._protein_protonator = ProtonateUtilsProteinProtonator()

    def _transform(self, structures: list[Structure]) -> list[Structure]:
        structures = self._select(structures)
        structures = self._prepare(structures)
        return self._protonate(structures)

    def _select(self, structures: list[Structure]) -> list[Structure]:
        volumes = [self._ligand_volume(s) for s in structures]
        chosen = int(np.argmax(volumes))
        return [structures[chosen], *(s for i, s in enumerate(structures) if i != chosen)]

    def _prepare(self, structures: list[Structure]) -> list[Structure]:
        chosen = structures[0]
        with tempfile.TemporaryDirectory() as tmp:
            protein_pdb = Path(tmp) / "protein.pdb"
            self._protein_atoms(chosen).write(str(protein_pdb))

            fixer = PDBFixer(filename=str(protein_pdb))
            fixer.findMissingResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()  # heavy atoms only; hydrogens come in _protonate

            fixed_pdb = Path(tmp) / "fixed.pdb"
            with open(fixed_pdb, "w") as fh:
                PDBFile.writeFile(fixer.topology, fixer.positions, fh)
            prepared = self._assemble(chosen, mda.Universe(str(fixed_pdb)).atoms, self._ligand_atoms(chosen))
        return [prepared, *structures[1:]]

    def _protonate(self, structures: list[Structure]) -> list[Structure]:
        # add ligand hydrogens first, while proteins still carry standard
        # residue names, then protonate the single canonical protein
        structures = [self._protonate_ligand(s) for s in structures]
        return [self._protonate_protein(structures[0]), *structures[1:]]

    def _protonate_protein(self, structure: Structure) -> Structure:
        with tempfile.TemporaryDirectory() as tmp:
            protein_pdb = Path(tmp) / "protein.pdb"
            self._protein_atoms(structure).write(str(protein_pdb))

            protonated_pdb = Path(tmp) / "protonated.pdb"
            protonated_pdb.write_text(self._protein_protonator.protonate(protein_pdb.read_text(), self._config.ph))
            return self._assemble(structure, mda.Universe(str(protonated_pdb)).atoms, self._ligand_atoms(structure))

    def _protonate_ligand(self, structure: Structure) -> Structure:
        mol = self._ligand_protonator.protonate(structure.to_rdkit_ligand_mol(selection=_LIGAND), self._config.ph)
        with tempfile.TemporaryDirectory() as tmp:
            ligand_pdb = Path(tmp) / "ligand.pdb"
            ligand_pdb.write_text(Chem.MolToPDBBlock(mol))
            ligand = mda.Universe(str(ligand_pdb)).atoms
            return self._assemble(structure, self._protein_atoms(structure), ligand)

    @staticmethod
    def _protein_atoms(structure: Structure) -> mda.AtomGroup:
        return structure.to_mda_universe().select_atoms("protein")

    @staticmethod
    def _ligand_atoms(structure: Structure) -> mda.AtomGroup:
        return structure.to_mda_universe().select_atoms("not protein")

    @classmethod
    def _ligand_volume(cls, structure: Structure) -> float:
        return float(ConvexHull(cls._ligand_atoms(structure).positions).volume)

    @staticmethod
    def _assemble(template: Structure, protein: mda.AtomGroup, ligand: mda.AtomGroup) -> Structure:
        """Build a Structure (PDB) from protein + ligand atoms, keeping metadata."""
        merged = mda.Merge(protein, ligand)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "merged.pdb"
            merged.atoms.write(str(out))
            data = base64.b64encode(out.read_bytes()).decode()
        return Structure(
            sequence=template.sequence,
            ligand_smiles=template.ligand_smiles,
            ligand_name=template.ligand_name,
            structure_format=StructureFormat.PDB,
            structure_data=data,
        )
