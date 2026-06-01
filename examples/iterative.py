#!/usr/bin/env python3

from pathlib import Path
import sys

from openplaceholder.impl.generator.openfold3 import OpenFold3Generator, OpenFold3GeneratorConfig

def get_run_openfold_path():
    import os

    if not (run_openfold_path := os.getenv("RUN_OPENFOLD3")):
        print("Missing \"RUN_OPENFOLD3\" environment variable.", file=sys.stderr)
        raise ValueError

    run_openfold_path = Path(run_openfold_path)

    if not run_openfold_path.exists():
        print(f"{run_openfold_path} does not exist.", file=sys.stderr)
        raise FileNotFoundError

    return run_openfold_path

def main() -> int:
    try:
        run_openfold_path = get_run_openfold_path()
    except Exception:
        return 1

    return 0

if __name__ == "__main__":
    return main()
