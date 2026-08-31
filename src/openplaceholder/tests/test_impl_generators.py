import tempfile

from openplaceholder.impl.generator.json import (
    JSONGenerator,
    JSONGeneratorConfig,
)
from openplaceholder.impl.generator.openfold3 import (
    OpenFold3Generator,
    OpenFold3GeneratorConfig,
)


class TestOpenFold3Generator:

    def test_init(self) -> None:

        with tempfile.TemporaryDirectory():
            config = OpenFold3GeneratorConfig(
                sequence="ABC",
                ligands={"methane": "C", "ethane": "CC", "propane": "CCC"},
                n_diffusion_samples=5,
                generate_n_seeds=5,
                generator_directory="./of3_dir/",
            )
            OpenFold3Generator(config)


class TestJSONGenerator:

    def test_init(self) -> None:

        with tempfile.NamedTemporaryFile() as f:
            config = JSONGeneratorConfig(path=f.name)
            JSONGenerator(config)
