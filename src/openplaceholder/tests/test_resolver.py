import tempfile
from pathlib import Path

from openplaceholder.core.resolver import load_toml, resolve_pipeline
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

            pipeline = resolve_pipeline(load_toml(config_path))

            generator = pipeline.generator
            assert isinstance(generator, OpenFold3Generator)
            assert isinstance(generator._config, OpenFold3GeneratorConfig)
            assert generator._config.sequence == "MGS"
            assert generator._config.ligands == {"lig_a": "C1=CC=CC=C1"}

            assert len(pipeline.validators) == 1
            assert isinstance(pipeline.validators[0], PosebustersValidator)

            selector = pipeline.selector
            assert isinstance(selector, MPOSelector)
            assert "VolumeOverlapObjective" in selector._config.objectives

            assert isinstance(pipeline.transformations, list)
            assert len(pipeline.transformations) == 3
            assert isinstance(pipeline.transformations[0], MaxVolumeSiteSubstitutionTransformation)
            assert isinstance(pipeline.transformations[1], HeavyAtomAdditionTransformation)
            protonation = pipeline.transformations[2]
            assert isinstance(protonation, ComplexProtonationTransformation)
            assert protonation._config.ph == 7.0
            assert isinstance(pipeline.mapping, LOMAPMapper)
