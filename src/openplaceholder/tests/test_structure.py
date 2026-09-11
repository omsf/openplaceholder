import base64

import MDAnalysis as mda
import pytest
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule

from openplaceholder.core.structure import (
    Structure,
    StructureFormat,
    UnsupportedFormatError,
)
from openplaceholder.tests.datafiles import TYK2_LIG_PDB
from openplaceholder.tests.helpers import make_structures, read_gzip_file

BENZENE_SMILES = "C1=CC=CC=C1"
PHENOL_SMILES = "C1=CC=C(C=C1)O"


def _tyk2_structure() -> Structure:
    content = base64.b64encode(read_gzip_file(str(TYK2_LIG_PDB))).decode()
    return Structure("SEQ", BENZENE_SMILES, "lig_ejm_55", "pdb", content)


def _hydrogenated_unl_structure(smiles: str, ligand_name: str) -> Structure:
    """A single-ligand PDB whose residue is named ``UNL`` (as the pipeline writes
    protonated ligands) and carries explicit hydrogens -- deliberately *not*
    named after ``ligand_name`` -- to exercise the default ligand selection."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    EmbedMolecule(mol, randomSeed=0xF00D)
    for atom in mol.GetAtoms():  # type: ignore
        info = Chem.AtomPDBResidueInfo()
        info.SetResidueName("UNL")
        info.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(info)
    block = base64.b64encode(Chem.MolToPDBBlock(mol).encode()).decode()
    return Structure("SEQ", smiles, ligand_name, "pdb", block)


class TestStructureFormat:

    def test_from_suffix_pdb(self) -> None:
        assert StructureFormat.from_suffix(".pdb") is StructureFormat.PDB

    def test_to_suffix_pdb(self) -> None:
        assert StructureFormat.PDB.to_suffix() == ".pdb"

    def test_from_suffix_case_insensitive(self) -> None:
        assert StructureFormat.from_suffix(".pDb") is StructureFormat.PDB

    def test_from_suffix_invalid(self) -> None:
        with pytest.raises(UnsupportedFormatError):
            StructureFormat.from_suffix(".ficmm")


class TestStructure:

    def test_valid_format_normalized(self) -> None:
        s = Structure("SEQ", BENZENE_SMILES, "lig", "pdb", "data")
        assert s.structure_format == StructureFormat.PDB

    def test_invalid_format(self) -> None:
        with pytest.raises(ValueError):
            Structure("SEQ", BENZENE_SMILES, "lig", "bdp", "data")

    def test_equality(self) -> None:
        a = Structure("SEQ", BENZENE_SMILES, "lig", "pdb", "data")
        b = Structure("SEQ", BENZENE_SMILES, "lig", "pdb", "data")
        c = Structure("OTHER", BENZENE_SMILES, "lig", "pdb", "data")
        assert a == b
        assert b != c

    def test_hashable(self) -> None:
        hash(Structure("SEQ", BENZENE_SMILES, "lig", "pdb", "data"))

    def test_structure_roundtrip(self) -> None:
        data = b"fake data"
        encoded = base64.b64encode(data).decode()
        s = Structure("SEQ", BENZENE_SMILES, "lig", "pdb", encoded)
        decoded = s.decode_structure_data()
        assert decoded == data, (data, decoded)

    def test_key_equal_structures(self) -> None:
        a = Structure("SEQ", BENZENE_SMILES, "lig", "pdb", "data")
        b = Structure("SEQ", BENZENE_SMILES, "lig", "pdb", "data")
        assert a.key == b.key

    def test_key_different_structures(self) -> None:
        a = Structure("SEQ", BENZENE_SMILES, "lig", "pdb", "data")
        b = Structure("SEQ", BENZENE_SMILES, "lig", "pdb", "different_data")
        assert a.key != b.key

    def test_to_mda_universe(self) -> None:
        test_file = str(TYK2_LIG_PDB)
        content = base64.b64encode(read_gzip_file(test_file)).decode()
        a = Structure(
            "TVFHKRYLKKIRDLGEGHFGKVSLYCYDPTNDGTGEMVAVKALKADCGPQHRSGWKQEIDILRTLYHEHIIKYKGCCEDQGEKSLQLVMEYVPLGSLRDYLPRHSIGLAQLLLFAQQICEGMAYLHAQHYIHRDLAARNVLLDNDRLVKIGDFGLAKAVPEGHEYYRVREDGDSPVFWYAPECLKEYKFYYASDVWSFGVTLYELLTHCDSSQSPPTKFLELIGIAQGQMTVLRLTELLERGERLPRPDKCPCEVYHLMKNCWETEASFRPTFENLIPILKTVHEKYQ",
            "[H]c1nc(N([H])C(=O)O[C@@]([H])([H])[H])c([H])c(N([H])C(=O)c2c(Cl)c([H])c([H])c([H])c2Cl)c1[H]",
            "lig_ejm_55",
            "pdb",
            content,
        )
        u = a.to_mda_universe()
        assert len(u.select_atoms("protein")) > len(u.select_atoms("not protein")) > 0

    def test_protein_and_ligand_atoms_partition_the_complex(self) -> None:
        s = _tyk2_structure()

        protein, ligand = s.protein_atoms(), s.ligand_atoms()

        assert len(protein) > len(ligand)
        assert len(ligand) > 0
        assert len(protein) + len(ligand) == len(s.to_mda_universe().atoms)

    def test_with_atoms_returns_pdb_copy_preserving_metadata(self) -> None:
        s = _tyk2_structure()
        merged = mda.Merge(s.protein_atoms(), s.ligand_atoms()).atoms

        rebuilt = s.with_atoms(merged)

        assert rebuilt.structure_format == StructureFormat.PDB
        assert rebuilt.sequence == s.sequence
        assert rebuilt.ligand_name == s.ligand_name
        assert len(rebuilt.to_mda_universe().atoms) == len(merged)

    def test_to_rdkit_ligand_mol_default_ignores_residue_name_and_hydrogens(self) -> None:
        # pipeline output names the ligand residue "UNL" (not ligand_name) and
        # carries explicit hydrogens; the default selection must perceive it
        # regardless -- a resname/hydrogen-sensitive default fails here.
        smiles = "CC(=O)Nc1ccccc1"
        s = _hydrogenated_unl_structure(smiles, ligand_name="acetanilide_007")

        mol = s.to_rdkit_ligand_mol()

        assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


@pytest.fixture
def structures() -> list[Structure]:
    structures = make_structures(n_structures=25, ligand_name="benzene")
    structures = structures + make_structures(n_structures=25, ligand_name="phenol", ligand_smiles=PHENOL_SMILES)
    return structures
