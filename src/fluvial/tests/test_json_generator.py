import tempfile

import pytest

from fluvial.core.structure import StructureSet
from fluvial.impl.generator.archiver import (
    JSONArchiver,
    JSONArchiverConfig,
)
from fluvial.impl.generator.json import (
    JSONGenerator,
    JSONGeneratorConfig,
)
from fluvial.tests.helpers import make_structures


class TestJSONGenerator:

    def test_invalid_path(self) -> None:
        with pytest.raises(FileNotFoundError):
            JSONGenerator(JSONGeneratorConfig(path="not_a_file.json"))

    def test_round_trip(self) -> None:
        with tempfile.NamedTemporaryFile() as tmpfile:
            artifact = StructureSet.from_structures([make_structures(n_structures=3)])

            archiver = JSONArchiver(JSONArchiverConfig(path=tmpfile.name))
            archiver.write(artifact)

            loaded = JSONGenerator(JSONGeneratorConfig(path=tmpfile.name)).run()
            assert len(loaded) == 1
            assert loaded == artifact
