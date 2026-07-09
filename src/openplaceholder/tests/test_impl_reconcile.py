from unittest import TestCase

import MDAnalysis as mda
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule

from openplaceholder.impl.protonation.reconcile import (
    ProlifInterfaceReconciler,
    _acceptor_idxs,
    _donor_hs,
)


def _amine_at(origin: list[float], seed: int) -> Chem.Mol:
    """A methylammonium (a pure H-bond donor: charged N, no lone pair) with its
    nitrogen translated to ``origin``."""
    mol = Chem.AddHs(Chem.MolFromSmiles("C[NH3+]"))
    EmbedMolecule(mol, randomSeed=seed)
    conf = mol.GetConformer()
    n_idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N")
    shift = np.array(origin) - np.array(conf.GetAtomPosition(n_idx))
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (np.array(conf.GetAtomPosition(i)) + shift).tolist())
    return mol


def _as_protein(mol: Chem.Mol) -> mda.AtomGroup:
    """Wrap an RDKit mol as an MDAnalysis AtomGroup carrying the resname/resid/
    name topology a PDB-derived protein would have. The nitrogen is named ``NZ``
    (a lysine-like sidechain donor) so it is not treated as a backbone amide."""
    universe = mda.Universe(mol)
    names = ["NZ" if a.element == "N" else f"{a.element}{i}" for i, a in enumerate(universe.atoms)]
    universe.add_TopologyAttr("name", names)
    universe.add_TopologyAttr("resname", ["UNK"] * len(universe.residues))
    universe.add_TopologyAttr("resid", [1] * len(universe.residues))
    return universe.atoms


class TestReconcileHelpers(TestCase):

    def test_donor_hs_finds_polar_hydrogen_donors(self) -> None:
        methylamine = Chem.AddHs(Chem.MolFromSmiles("CN"))
        # the amine nitrogen is the only donor; the methyl carbon is not
        self.assertEqual(len(_donor_hs(methylamine)), 1)

    def test_acceptor_idxs_finds_lone_pair_acceptors(self) -> None:
        formaldehyde = Chem.AddHs(Chem.MolFromSmiles("C=O"))
        self.assertEqual(len(_acceptor_idxs(formaldehyde)), 1)


class TestProlifInterfaceReconciler(TestCase):

    def test_removes_hydrogen_from_donor_donor_clash(self) -> None:
        protein = _as_protein(_amine_at([0.0, 0.0, 0.0], seed=1))
        ligand = _amine_at([2.8, 0.0, 0.0], seed=2)  # N..N within H-bond range

        fixed = ProlifInterfaceReconciler().reconcile(protein, ligand)

        self.assertEqual(len(fixed), len(protein) - 1)

    def test_leaves_non_clashing_interface_untouched(self) -> None:
        protein = _as_protein(_amine_at([0.0, 0.0, 0.0], seed=1))
        ligand = _amine_at([15.0, 0.0, 0.0], seed=2)  # far apart, no contact

        fixed = ProlifInterfaceReconciler().reconcile(protein, ligand)

        self.assertEqual(len(fixed), len(protein))
