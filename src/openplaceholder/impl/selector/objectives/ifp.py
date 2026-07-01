"""Interaction-fingerprint (IFP) similarity objective: rewards pairs of
ligand poses whose predicted binding modes make the same protein contacts."""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import MDAnalysis as mda
from prolif.fingerprint import Fingerprint
from prolif.molecule import Molecule
from rdkit import Chem

from openplaceholder.core.selection.objective import Objective, ObjectiveConfig
from openplaceholder.core.structure import Structure


@dataclass(frozen=True, eq=True)
class IFPSimilarityObjectiveConfig(ObjectiveConfig):
    # which ProLIF interaction types to consider; None uses all of them
    interactions: list[str] | None = None


class IFPSimilarityObjective(Objective):
    """Jaccard similarity between two poses' ProLIF interaction fingerprints.

    Each side's fingerprint is generated independently against its own
    predicted protein conformation (each Structure is a full co-folded
    complex, so the two poses generally don't share one). Similarity is
    therefore computed over the *sets* of (protein residue, interaction
    type) contacts each pose makes, rather than over fixed-width bit
    vectors -- those are only comparable when both fingerprints come from
    one shared run (e.g. one fixed protein, many ligands), which doesn't
    hold here.
    """

    _config: IFPSimilarityObjectiveConfig

    def __init__(self, config: IFPSimilarityObjectiveConfig):
        super().__init__(config)
        # matrix() calls score() once per *pair*, but each side's contact set
        # only depends on that one structure; caching keeps the expensive
        # PDB-round-trip + ProLIF fingerprinting to one call per structure
        # (O(n)) rather than one per pair (O(n^2)) -- the difference between
        # tractable and not once the pool gets into the hundreds/thousands.
        self._contacts_cache: dict[Structure, set[tuple[str, str]]] = {}

    def score(self, a: Structure, b: Structure) -> float:
        contacts_a = self._contacts(a)
        contacts_b = self._contacts(b)
        if not contacts_a and not contacts_b:
            return 0.0
        return len(contacts_a & contacts_b) / len(contacts_a | contacts_b)

    def _contacts(self, structure: Structure) -> set[tuple[str, str]]:
        if structure not in self._contacts_cache:
            self._contacts_cache[structure] = self._compute_contacts(structure)
        return self._contacts_cache[structure]

    def _compute_contacts(self, structure: Structure) -> set[tuple[str, str]]:
        ligand = Molecule.from_rdkit(structure.to_rdkit_ligand_mol())
        protein = Molecule.from_rdkit(self._protein_mol(structure))

        fp = Fingerprint(interactions=self._config.interactions or "all")
        ifp = fp.generate(ligand, protein)

        contacts: set[tuple[str, str]] = set()
        for (_, protein_residue), present in ifp.items():
            for interaction, is_present in zip(fp.interactions, present):
                if is_present:
                    contacts.add((str(protein_residue), interaction))
        return contacts

    @staticmethod
    def _protein_mol(structure: Structure) -> Chem.Mol:
        """Build an RDKit Mol for the protein via a PDB round-trip.

        Unlike the ligand (whose correct connectivity is known from its
        SMILES template), the protein's bonds/aromaticity have to be
        perceived from its 3D coordinates. RDKit's own PDB parser does this
        far more robustly than MDAnalysis's name-keyed bond guesser (which
        fails outright on this data -- see Structure.to_rdkit_ligand_mol),
        since it recognizes standard residue/atom naming directly.
        """
        protein_atoms = structure.to_mda_universe().select_atoms("protein")

        with tempfile.TemporaryDirectory() as tmp:
            pdb_path = Path(tmp) / "protein.pdb"
            with mda.Writer(str(pdb_path), n_atoms=len(protein_atoms)) as writer:
                writer.write(protein_atoms)
            # the MMCIF parser's default altLoc value is a literal NUL byte,
            # which corrupts the fixed-width PDB columns MDAnalysis writes it
            # into and breaks RDKit's PDB parser after the very first atom.
            pdb_block = pdb_path.read_text().replace("\x00", " ")

        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False, proximityBonding=True)
        # predicted (not crystallographic) coordinates can have minor local
        # geometry issues that trip strict valence checks; aromaticity and
        # other perception still matters for interaction typing, so sanitize
        # everything except that one check rather than failing outright.
        relaxed = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
        Chem.SanitizeMol(mol, sanitizeOps=relaxed, catchErrors=True)
        return mol
