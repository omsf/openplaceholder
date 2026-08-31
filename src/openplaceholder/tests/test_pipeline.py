import tempfile
from pathlib import Path

from openplaceholder.core.loader import load_toml
from openplaceholder.core.pipeline import Pipeline
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


class TestResolvePipeline:

    def test_resolve_pipeline_from_toml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(CONFIG_TEMPLATE.format(generator_directory=tmpdir))

            pipeline = Pipeline.from_config_map(load_toml(config_path))

            generator = pipeline.plugins[0]
            assert isinstance(generator, OpenFold3Generator)
            assert isinstance(generator._config, OpenFold3GeneratorConfig)
            assert generator._config.sequence == "MGS"
            assert generator._config.ligands == {"lig_a": "C1=CC=CC=C1"}

            assert isinstance(pipeline.plugins[1], PosebustersValidator)

            selector = pipeline.plugins[2]
            assert isinstance(selector, MPOSelector)
            assert "VolumeOverlapObjective" in selector._config.objectives

            assert isinstance(pipeline.plugins[3], MaxVolumeSiteSubstitutionTransformation)
            assert isinstance(pipeline.plugins[4], HeavyAtomAdditionTransformation)
            protonation = pipeline.plugins[5]
            assert isinstance(protonation, ComplexProtonationTransformation)
            assert protonation._config.ph == 7.0

            assert isinstance(pipeline.plugins[6], LOMAPMapper)
