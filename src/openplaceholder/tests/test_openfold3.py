import tempfile
from pathlib import Path
from unittest.mock import patch

from openplaceholder.impl.generator.openfold3 import (
    OpenFold3Generator,
    OpenFold3GeneratorConfig,
)


def generate_fake_run_output(gen_dir: Path, n_seeds: int, n_diffusion_samples: int, ligands: dict[str, str]) -> None:
    import random

    padded = lambda s: f"{s:0>2}"
    padded_range = lambda n: map(padded, range(1, n + 1))

    for lig, _ in ligands.items():
        for seed_num in padded_range(n_seeds):  # type: ignore
            seed = f"seed_{seed_num}"
            for pose_num in padded_range(n_diffusion_samples):  # type: ignore
                basename = f"pose_{pose_num}.pdb"
                output_dir = gen_dir / "output" / lig / seed
                output_dir.mkdir(exist_ok=True, parents=True)
                file_path = output_dir / basename
                file_path.write_bytes(random.randbytes(8))


class TestOpenFold3Generator:

    def test_mocked_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OpenFold3GeneratorConfig(
                sequence="ABC",
                ligands={"lig_01": "C", "lig_02": "CC", "lig_03": "CCC"},
                n_diffusion_samples=5,
                generate_n_seeds=5,
                generator_directory=tmpdir,
                run_openfold_path="nope",
            )
            with patch.object(OpenFold3Generator, "_run_openfold_subprocess", autospec=True) as mock_run_subprocess:
                mock_run_subprocess.return_value = None
                gen = OpenFold3Generator(config)

                assert isinstance(gen._config.generate_n_seeds, int)

                generate_fake_run_output(
                    Path(gen._config.generator_directory),
                    gen._config.generate_n_seeds,
                    gen._config.n_diffusion_samples,
                    gen._config.ligands,
                )
                artifacts = gen.run()

            assert len(artifacts) == 3

            for artifact in artifacts:
                assert len(artifact.structures) == gen._config.generate_n_seeds * gen._config.n_diffusion_samples
