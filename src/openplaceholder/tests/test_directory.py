import tempfile
from unittest import TestCase

from openplaceholder.core.generation.archive import (
    DirectoryArchive,
    DirectoryArchiveConfig,
)
from openplaceholder.core.generation.generator import StructureGeneratorArtifact
from openplaceholder.impl.generator.directory import (
    DirectoryGenerator,
    DirectoryGeneratorConfig,
)
from openplaceholder.tests.helpers import make_structures


class TestDirectoryGenerator(TestCase):

    def test_invalid_directory(self) -> None:
        with self.assertRaises(FileNotFoundError):
            DirectoryGenerator(DirectoryGeneratorConfig("not_a_directory/"))

    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = StructureGeneratorArtifact.from_structures(make_structures(n_structures=3))

            archive = DirectoryArchive(DirectoryArchiveConfig(directory=tmpdir))
            archive.write([artifact])

            loaded = DirectoryGenerator(DirectoryGeneratorConfig(tmpdir)).run()
            assert len(loaded) == 1
            assert loaded[0].sequence == artifact.sequence
            assert loaded[0].ligand_smiles == artifact.ligand_smiles
            assert loaded[0].ligand_name == artifact.ligand_name
            assert loaded[0].structures == artifact.structures
