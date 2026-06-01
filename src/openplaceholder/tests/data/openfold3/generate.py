import random
import pathlib

output_dir = pathlib.Path() / "sample_output"
padded = lambda s: f"{s:0>2}"

for lig_num in map(padded, range(1, 21)):

    lig = f"lig_{lig_num}"

    for seed_num in map(padded, range(1, 6)):

        seed = f"seed_{seed_num}"

        for pose_num in map(padded, range(1, 6)):

            basename = f"pose_{pose_num}.cif"
            output_dir = cwd / lig / seed
            output_dir.mkdir(exist_ok=True, parents=True)
            file_path = output_dir / basename

            file_path.write_bytes(random.randbytes(8))

