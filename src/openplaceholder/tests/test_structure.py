import base64
import json
import tempfile
from pathlib import Path
from unittest import TestCase, skip

from openplaceholder.core.structure.structure import (
    Structure,
    StructureFormat,
    StructureSet,
)
from openplaceholder.tests.helpers import make_structures

BENZENE_SMILES = "C1=CC=CC=C1"
PHENOL_SMILES = "C1=CC=C(C=C1)O"


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

    @skip("")
    def test_same_complex(self) -> None:
        raise NotImplementedError


class TestStructureSet(TestCase):

    def setUp(self) -> None:
        self.structures = make_structures(n_structures=25, ligand_name="benzene")
        self.structures += make_structures(n_structures=25, ligand_name="phenol", ligand_smiles=PHENOL_SMILES)

    def test_from_structures(self) -> None:
        ss_one_to_one = StructureSet.from_structures(self.structures)
        assert len(ss_one_to_one) == len(self.structures)

        ss_doubled_input = StructureSet.from_structures(self.structures + self.structures)
        assert len(ss_doubled_input) == len(self.structures)

    def test_write(self) -> None:
        ss = StructureSet.from_structures(self.structures)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "structures.json"
            ss.write(path)
            assert path.exists()
            content = json.loads(path.read_text())
            assert "structures" in content
            assert len(content["structures"]) == len(self.structures), len(content["structures"])

    def test_from_file(self) -> None:
        ss = StructureSet.from_structures(self.structures)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "structures.json"
            ss.write(path)
            loaded = StructureSet.from_file(path)
            assert len(loaded) == len(self.structures)
            loaded_deduped = StructureSet.from_structures(loaded.structures)
            assert len(loaded_deduped) == len(loaded)
