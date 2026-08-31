#!/usr/bin/env python3

import logging
from pathlib import Path

from openplaceholder.core.loader import load_toml
from openplaceholder.core.pipeline import Pipeline
from openplaceholder.core.runner import run_serial

logging.basicConfig(level=logging.INFO)

TOML_CONFIG = Path(__file__).parents[1] / "config.toml"

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    pipeline = Pipeline.from_config_map(load_toml(TOML_CONFIG))
    result = run_serial(pipeline, None)
    result.to_json("alchemicalnetwork.json")
