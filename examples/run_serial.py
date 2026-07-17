#!/usr/bin/env python3

import logging
from pathlib import Path
import logging

from openplaceholder.core.resolver import resolve_pipeline
from openplaceholder.core.runner import run_serial

logging.basicConfig(level=logging.INFO)

TOML_CONFIG = Path(__file__).parents[1] / "config.toml"

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    pipeline = resolve_pipeline(TOML_CONFIG)
    result = run_serial(pipeline)
    # since mappers are not implemented, the following code is unreachable for now
    result.to_json("alchemicalnetwork.json")
