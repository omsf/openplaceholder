"""prolif-guided reconciliation of protein/ligand H-bond mismatches.

Protein and ligand are protonated independently (neither protonator sees the
other), which leaves two failure modes at the interface:

  * donor-donor       -- both polar atoms carry an H pointing at each other: a
                         steric clash. Fixed here by removing the protein-side
                         H, turning that donor into an acceptor.
  * acceptor-acceptor -- neither carries an H: a *missing* H-bond, not a clash.
                         Correcting it would mean inventing a proton (and a
                         formal charge), so it is only warned about.

prolif supplies the donor/acceptor SMARTS (the chemistry); the geometry and the
edit are plain numpy/MDAnalysis. This pass only ever *removes* hydrogens, never
adds them.
"""

import logging

import MDAnalysis as mda
import numpy as np
from prolif.interactions import HBDonor  # type: ignore[attr-defined]
from prolif.molecule import Molecule
from rdkit import Chem

logger = logging.getLogger(__name__)

# Reuse prolif's own H-bond definitions rather than hand-rolling SMARTS. Both
# roles live on the HBDonor interaction; pick by atom count so we do not depend
# on prolif's lig_/prot_ naming: the donor pattern is the two-atom "[D]-[H]",
# the acceptor pattern the single lone-pair atom.
_PATTERNS = (HBDonor().lig_pattern, HBDonor().prot_pattern)
_DONOR: Chem.Mol = next(m for m in _PATTERNS if m.GetNumAtoms() == 2)
_ACCEPTOR: Chem.Mol = next(m for m in _PATTERNS if m.GetNumAtoms() == 1)

_HB_CUT = 3.5  # heavy-heavy donor/acceptor contact distance (Angstrom)
_BACKBONE = {"N"}  # peptide amide nitrogen -- not titratable, cannot deprotonate


def _donor_hs(mol: Chem.Mol) -> dict[int, list[int]]:
    """Map each heavy-donor atom index to the indices of its donor hydrogens."""
    out: dict[int, list[int]] = {}
    for heavy, hydrogen in mol.GetSubstructMatches(_DONOR):
        out.setdefault(heavy, []).append(hydrogen)
    return out


def _acceptor_idxs(mol: Chem.Mol) -> set[int]:
    """Heavy-atom indices that can accept a hydrogen bond."""
    return {match[0] for match in mol.GetSubstructMatches(_ACCEPTOR)}


class ProlifInterfaceReconciler:
    """Deprotonate protein donors that clash with ligand donors; warn on
    acceptor-acceptor misses.

    ``Molecule.from_mda`` preserves atom order, so RDKit atom indices align
    one-to-one with the input protein ``AtomGroup``. ``ligand_mol`` must carry
    explicit hydrogens and a conformer (i.e. the *protonated* ligand).
    """

    def reconcile(self, protein_ag: mda.AtomGroup, ligand_mol: Chem.Mol) -> mda.AtomGroup:
        protein = Molecule.from_mda(protein_ag)
        ligand = Molecule(ligand_mol)
        p_xyz = protein.GetConformer().GetPositions()
        l_xyz = ligand.GetConformer().GetPositions()

        p_donors, l_donors = _donor_hs(protein), _donor_hs(ligand)
        p_acceptors, l_acceptors = _acceptor_idxs(protein), _acceptor_idxs(ligand)

        drop: set[int] = set()  # protein hydrogen indices to delete
        for p_heavy, p_hs in p_donors.items():
            for l_heavy in l_donors:
                if np.linalg.norm(p_xyz[p_heavy] - l_xyz[l_heavy]) > _HB_CUT:
                    continue
                if p_heavy in p_acceptors or l_heavy in l_acceptors:
                    continue  # either side can accept -> complementary, not a clash
                atom = protein_ag[p_heavy]
                if atom.name in _BACKBONE:
                    logger.warning(
                        "donor-donor clash at backbone %s%d:%s -- left as-is (peptide N-H not titratable)",
                        atom.resname,
                        atom.resid,
                        atom.name,
                    )
                    continue
                clashing_h = min(p_hs, key=lambda i: float(np.linalg.norm(p_xyz[i] - l_xyz[l_heavy])))
                drop.add(clashing_h)
                logger.info("donor-donor clash: deprotonating %s%d:%s", atom.resname, atom.resid, atom.name)

        for p_heavy in p_acceptors - set(p_donors):  # pure protein acceptors only
            for l_heavy in l_acceptors - set(l_donors):
                if np.linalg.norm(p_xyz[p_heavy] - l_xyz[l_heavy]) <= _HB_CUT:
                    atom = protein_ag[p_heavy]
                    logger.warning(
                        "acceptor-acceptor contact (missing H-bond) at %s%d:%s -- review / Protoss",
                        atom.resname,
                        atom.resid,
                        atom.name,
                    )

        if not drop:
            return protein_ag
        return protein_ag[[i for i in range(len(protein_ag)) if i not in drop]]
