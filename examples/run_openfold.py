#!/usr/bin/env python3

import os
import random
from pathlib import Path
import tomllib
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("run_openfold.py")

from openplaceholder.impl.generator.openfold3 import (
    OpenFold3Generator,
    OpenFold3GeneratorConfig,
)
from openplaceholder.impl.generator.archiver import JSONArchiver, JSONArchiverConfig

TOML_CONFIG = Path(__file__).parents[1] / "config.toml"

if __name__ == "__main__":

    if not TOML_CONFIG.exists():
        raise FileNotFoundError(f"Could not find {TOML_CONFIG}")

    config = tomllib.loads(TOML_CONFIG.read_text())
    generator_config = config["generation"]["generator"]

    # normally this would be set in the config directly, but that is
    # impractical for this example
    generator_config["run_openfold_path"] = os.getenv("RUN_OPENFOLD_PATH")

    if gendir := os.getenv("GEN_DIR_OVERRIDE"):
        logging.info("Overriding generator directory with %s", gendir)
        generator_config["generator_directory"] = gendir

    if not generator_config["implementation"] == "openplaceholder.impl.generator.openfold3:OpenFold3Generator":
        raise ValueError("This script is only configured to run the OpenFold3Generator")


    # in theory you could double splat these into the
    # OpenFold3Generator, but it's explicitly unpacked here for
    # demonstration.
    num_seeds = generator_config["generate_n_seeds"]
    n_diffusion_samples = generator_config["n_diffusion_samples"]
    gen_dir = Path(generator_config["generator_directory"])
    clean_up = generator_config.get("clean_up") or False
    sequence = generator_config["sequence"]
    ligands = generator_config["ligands"]
    run_openfold_path = os.getenv("RUN_OPENFOLD_PATH")

    # set up an archiver to grab the packaged structures
    archiver_config = JSONArchiverConfig(path="./archive.json")
    archiver = JSONArchiver(archiver_config)

    config = OpenFold3GeneratorConfig(
        sequence=sequence,
        ligands=ligands,
        n_diffusion_samples=n_diffusion_samples,
        generator_directory=gen_dir,
        run_openfold_path=run_openfold_path,
        generate_n_seeds=num_seeds,
        clean_up=clean_up,
    )

    if run_openfold_path is None:
        raise ValueError("Missing RUN_OPENFOLD_PATH")

    generator = OpenFold3Generator(config)
    structures = generator.run()

    # write structures to disk
    archiver.write(structures)
