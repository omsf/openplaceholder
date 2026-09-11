import tempfile

from openplaceholder.core.structure import StructureSet
from openplaceholder.impl.generator.archiver import JSONArchiver, JSONArchiverConfig
from openplaceholder.tests.helpers import make_structures


class TestJSONArchiver:

    def test_roundtrip(self) -> None:

        with tempfile.NamedTemporaryFile() as tmpfile:
            artifact = StructureSet.from_structures([make_structures(n_structures=3)])
            archiver = JSONArchiver(JSONArchiverConfig(path=tmpfile.name))
            archiver.write(artifact)
            loaded = archiver.read()

            assert len(loaded) == 1
            assert loaded.replicate_sets == artifact.replicate_sets
