import base64
from unittest import TestCase

from openplaceholder.core.structure.structure import Structure, StructureFormat

BENZENE_SMILES = "C1=CC=CC=C1"


class TestStructureFormat(TestCase):

    def test_from_suffix_mmcif(self) -> None:
        assert StructureFormat.from_suffix(".mmcif") is StructureFormat.MMCIF

    def test_from_suffix_pdb(self) -> None:
        assert StructureFormat.from_suffix(".pdb") is StructureFormat.PDB

    def test_to_suffix_pdb(self) -> None:
        assert StructureFormat.PDB.to_suffix() == ".pdb"

    def test_to_suffix_mmcif(self) -> None:
        assert StructureFormat.MMCIF.to_suffix() == ".mmcif"

    def test_from_suffix_case_insensitive(self) -> None:
        assert StructureFormat.from_suffix(".MMCIF") is StructureFormat.MMCIF

    def test_from_suffix_invalid(self) -> None:
        with self.assertRaises(ValueError):
            StructureFormat.from_suffix(".ficmm")


class TestStructure(TestCase):

    def test_valid_format_normalized(self) -> None:
        s = Structure("SEQ", BENZENE_SMILES, "lig", "mmcif", "data")
        assert s.structure_format == "MMCIF"

    def test_invalid_format(self) -> None:
        with self.assertRaises(ValueError):
            Structure("SEQ", BENZENE_SMILES, "lig", "ficmm", "data")

    def test_equality(self) -> None:
        a = Structure("SEQ", BENZENE_SMILES, "lig", "mmcif", "data")
        b = Structure("SEQ", BENZENE_SMILES, "lig", "mmcif", "data")
        c = Structure("OTHER", BENZENE_SMILES, "lig", "mmcif", "data")
        d = Structure("OTHER", BENZENE_SMILES, "lig", "pdb", "data")
        assert a == b
        assert b != c
        assert c != d

    def test_hashable(self) -> None:
        hash(Structure("SEQ", BENZENE_SMILES, "lig", "mmcif", "data"))

    def test_structure_roundtrip(self) -> None:
        data = b"fake data"
        encoded = base64.b64encode(data).decode()
        s = Structure("SEQ", BENZENE_SMILES, "lig", "mmcif", encoded)
        assert s.structure == data

    def test_key_equal_structures(self) -> None:
        a = Structure("SEQ", BENZENE_SMILES, "lig", "mmcif", "data")
        b = Structure("SEQ", BENZENE_SMILES, "lig", "mmcif", "data")
        assert a.key == b.key

    def test_key_different_structures(self) -> None:
        a = Structure("SEQ", BENZENE_SMILES, "lig", "mmcif", "data")
        b = Structure("SEQ", BENZENE_SMILES, "lig", "mmcif", "different_data")
        assert a.key != b.key
