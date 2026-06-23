import tempfile
from unittest import TestCase

from openplaceholder.core.generation.generator import StructureGeneratorArtifact
from openplaceholder.impl.generator.archiver import (
    JSONArchiver,
    JSONArchiverConfig,
)
from openplaceholder.impl.generator.json import (
    JSONGenerator,
    JSONGeneratorConfig,
)
from openplaceholder.tests.helpers import make_structures


class TestJSONGenerator(TestCase):

    def test_invalid_path(self) -> None:
        with self.assertRaises(FileNotFoundError):
            JSONGenerator(JSONGeneratorConfig(path="not_a_file.json"))

    def test_round_trip(self) -> None:
        with tempfile.NamedTemporaryFile() as tmpfile:
            artifact = StructureGeneratorArtifact.from_structures(make_structures(n_structures=3))

            archiver = JSONArchiver(JSONArchiverConfig(path=tmpfile.name))
            archiver.write([artifact])

            loaded = JSONGenerator(JSONGeneratorConfig(path=tmpfile.name)).run()
            assert len(loaded) == 1
            assert loaded[0].sequence == artifact.sequence
            assert loaded[0].ligand_smiles == artifact.ligand_smiles
            assert loaded[0].ligand_name == artifact.ligand_name
            assert loaded[0].structures == artifact.structures
