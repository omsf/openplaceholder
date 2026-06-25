import tempfile
from unittest import TestCase

from openplaceholder.impl.generator.directory import (
    DirectoryGenerator,
    DirectoryGeneratorConfig,
)
from openplaceholder.impl.generator.json import (
    JSONGenerator,
    JSONGeneratorConfig,
)
from openplaceholder.impl.generator.openfold3 import (
    OpenFold3Generator,
    OpenFold3GeneratorConfig,
)


class TestOpenFold3Generator(TestCase):

    def test_init(self) -> None:

        with tempfile.TemporaryDirectory() as d:
            config = OpenFold3GeneratorConfig(
                sequence="ABC",
                ligands={"methane": "C", "ethane": "CC", "propane": "CCC"},
                n_diffusion_samples=5,
                generate_n_seeds=5,
                generator_directory="./of3_dir/",
            )
            OpenFold3Generator(config)


class TestDirectoryGenerator(TestCase):

    def test_init(self) -> None:

        with tempfile.TemporaryDirectory() as d:
            config = DirectoryGeneratorConfig(path=str(d))
            DirectoryGenerator(config)


class TestJSONGenerator(TestCase):

    def test_init(self) -> None:

        with tempfile.NamedTemporaryFile() as f:
            config = JSONGeneratorConfig(path=f.name)
            JSONGenerator(config)
