import tempfile
from pathlib import Path
from unittest import TestCase

from openplaceholder.core.resolver import resolve_pipeline
from openplaceholder.impl.generator.openfold3 import (
    OpenFold3Generator,
    OpenFold3GeneratorConfig,
)
from openplaceholder.impl.mappers import LOMAPMapper
from openplaceholder.impl.selector.mpo import MPOSelector
from openplaceholder.impl.transformations import (
    ComplexProtonationTransformation,
    HeavyAtomAdditionTransformation,
    MaxVolumeSiteSubstitutionTransformation,
)
from openplaceholder.impl.validators import PosebustersValidator

CONFIG_TEMPLATE = """
[generation.generator]
implementation = "openplaceholder.impl.generator.openfold3:OpenFold3Generator"
n_diffusion_samples = 5
generate_n_seeds = 5
generator_directory = "{generator_directory}"
sequence = "MGS"

[generation.generator.ligands]
lig_a = "C1=CC=CC=C1"

[[selection.validators]]
implementation = "openplaceholder.impl.validators:PosebustersValidator"

[selection.selector]
implementation = "openplaceholder.impl.selector.mpo:MPOSelector"

[selection.selector.objectives]
VolumeOverlapObjective = {{ weight = 1.0 }}

[[assembly.transformations]]
implementation = "openplaceholder.impl.transformations:MaxVolumeSiteSubstitutionTransformation"

[[assembly.transformations]]
implementation = "openplaceholder.impl.transformations:HeavyAtomAdditionTransformation"

[[assembly.transformations]]
implementation = "openplaceholder.impl.transformations:ComplexProtonationTransformation"
ph = 7.0

[assembly.mapping]
implementation = "openplaceholder.impl.mappers:LOMAPMapper"
"""


class TestResolvePipeline(TestCase):

    def test_resolve_pipeline_from_toml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(CONFIG_TEMPLATE.format(generator_directory=tmpdir))

            pipeline = resolve_pipeline(config_path)

            generator = pipeline.generator
            assert isinstance(generator, OpenFold3Generator)
            assert isinstance(generator._config, OpenFold3GeneratorConfig)
            self.assertEqual(generator._config.sequence, "MGS")
            self.assertEqual(generator._config.ligands, {"lig_a": "C1=CC=CC=C1"})

            self.assertEqual(len(pipeline.validators), 1)
            self.assertIsInstance(pipeline.validators[0], PosebustersValidator)

            selector = pipeline.selector
            assert isinstance(selector, MPOSelector)
            self.assertIn("VolumeOverlapObjective", selector._config.objectives)

            self.assertIsInstance(pipeline.transformations, list)
            self.assertEqual(len(pipeline.transformations), 3)
            self.assertIsInstance(pipeline.transformations[0], MaxVolumeSiteSubstitutionTransformation)
            self.assertIsInstance(pipeline.transformations[1], HeavyAtomAdditionTransformation)
            protonation = pipeline.transformations[2]
            assert isinstance(protonation, ComplexProtonationTransformation)
            self.assertEqual(protonation._config.ph, 7.0)
            self.assertIsInstance(pipeline.mapping, LOMAPMapper)
