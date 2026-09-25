#!/usr/bin/env python3

import logging
from pathlib import Path

from openplaceholder.core.loader import load_toml
from openplaceholder.core.pipeline import Pipeline
from openplaceholder.core.runner import run_prefect
from openplaceholder.core.structure import StructureSet

logging.basicConfig(level=logging.INFO)

TOML_CONFIG = Path(__file__).parents[1] / "config.toml"

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    initial_data: StructureSet = StructureSet.from_json(Path("results.json"))
    pipeline = Pipeline.from_config_map(load_toml(TOML_CONFIG), allow_partial=True)
    result = run_prefect(pipeline, initial_data)
    result.to_json("alchemicalnetwork.json")
