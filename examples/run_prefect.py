#!/usr/bin/env python3

import logging
from pathlib import Path

from gufe.network import AlchemicalNetwork
from gufe.tokenization import GufeKey

from openplaceholder.core.io import DataRef
from openplaceholder.core.loader import load_toml

from openplaceholder.core.orchestration.flow import run_prefect_ray
from openplaceholder.core.pipeline import Pipeline
from openplaceholder.core.structure import StructureSet

logging.basicConfig(level=logging.INFO)

TOML_CONFIG = Path(__file__).parents[1] / "config.toml"

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    data_ref: DataRef[StructureSet] = DataRef(Path("./results.json").resolve().as_uri(), GufeKey(""), StructureSet)
    data_ref: DataRef[AlchemicalNetwork] = DataRef(
        Path("./alchemical_network.json").resolve().as_uri(), GufeKey(""), AlchemicalNetwork
    )

    pipeline = Pipeline.from_config_map(load_toml(TOML_CONFIG), allow_partial=True)
    result: DataRef = run_prefect_ray(pipeline, data_ref)
