import tempfile
from unittest import TestCase

from openplaceholder.core.generation.archive import JSONArchive, JSONArchiveConfig
from openplaceholder.core.generation.generator import StructureGeneratorArtifact
from openplaceholder.tests.helpers import make_structures


class TestJSONArchive(TestCase):

    def test_roundtrip(self) -> None:

        with tempfile.NamedTemporaryFile() as tmpfile:
            artifact = StructureGeneratorArtifact.from_structures(make_structures(n_structures=3))
            archive = JSONArchive(JSONArchiveConfig(path=tmpfile.name))
            archive.write([artifact])
            loaded = archive.read()

            assert len(loaded) == 1
            assert loaded[0].sequence == artifact.sequence
            assert loaded[0].ligand_smiles == artifact.ligand_smiles
            assert loaded[0].ligand_name == artifact.ligand_name
            assert loaded[0].structures == artifact.structures
