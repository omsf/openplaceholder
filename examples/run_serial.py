#!/usr/bin/env python3

import logging
from pathlib import Path

from openplaceholder.core.resolver import load_toml, resolve_pipeline
from openplaceholder.core.runner import run_serial

logging.basicConfig(level=logging.INFO)

TOML_CONFIG = Path(__file__).parents[1] / "config.toml"

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    pipeline = resolve_pipeline(load_toml(TOML_CONFIG))
    result = run_serial(pipeline)
    result.to_json("alchemicalnetwork.json")
