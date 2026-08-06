import tempfile

from openplaceholder.core.generation.generator import StructureGeneratorArtifact
from openplaceholder.impl.generator.archiver import JSONArchiver, JSONArchiverConfig
from openplaceholder.tests.helpers import make_structures


class TestJSONArchiver:

    def test_roundtrip(self) -> None:

        with tempfile.NamedTemporaryFile() as tmpfile:
            artifact = StructureGeneratorArtifact.from_structures(make_structures(n_structures=3))
            archiver = JSONArchiver(JSONArchiverConfig(path=tmpfile.name))
            archiver.write([artifact])
            loaded = archiver.read()

            assert len(loaded) == 1
            assert loaded[0].sequence == artifact.sequence
            assert loaded[0].ligand_smiles == artifact.ligand_smiles
            assert loaded[0].ligand_name == artifact.ligand_name
            assert loaded[0].structures == artifact.structures
