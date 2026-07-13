import os
from pathlib import Path

from openplaceholder.core.resolver import resolve_pipeline
from openplaceholder.core.runner import run_serial

TOML_CONFIG = Path(__file__).parents[1] / "config.toml"

if __name__ == "__main__":
    pipeline = resolve_pipeline(TOML_CONFIG)
    result = run_serial(pipeline)
    # since mappers are not implemented, the following code is unreachable for now
    results.to_json("alchemicalnetwork.json")
