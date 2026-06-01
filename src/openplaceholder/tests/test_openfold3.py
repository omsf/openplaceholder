import tempfile
import unittest


def generate_fake_run_output(out_dir: Path):
    import pathlib
    import random

    output_dir = pathlib.Path() / "sample_output"
    padded = lambda s: f"{s:0>2}"
    padded_range = lambda n: map(padded, range(1, n + 1))

    for lig_num in padded_range(20):
        lig = f"lig_{lig_num}"
        for seed_num in padded_range(5):
            seed = f"seed_{seed_num}"
            for pose_num in padded_range(5):
                basename = f"pose_{pose_num}.cif"
                output_dir = cwd / lig / seed
                output_dir.mkdir(exist_ok=True, parents=True)
                file_path = output_dir / basename
                file_path.write_bytes(random.randbytes(8))


class TestOpenFold3Configuration(unittest.TestCase):

    def test_defaults(self) -> None:
        pass


class TestOpenFold3(unittest.TestCase):

    def setUp(self) -> None:
        self.skipTest("Not testing this yet")

    def test_import_openfold3_generator(self) -> None:
        raise ImportError
