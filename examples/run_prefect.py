#!/usr/bin/env python3

import logging
from pathlib import Path

from gufe.tokenization import GufeKey

from fluvial.core.io import DataRef
from fluvial.core.loader import load_toml
from fluvial.core.orchestration.flow import run_prefect_ray
from fluvial.core.pipeline import Pipeline, Stage
from fluvial.core.structure import StructureSeries

logging.basicConfig(level=logging.INFO)
TOML_CONFIG = Path(__file__).parents[1] / "config.toml"

if __name__ == "__main__":

    data_ref: DataRef[StructureSeries] = DataRef(
        Path("./transformed.json").resolve().as_uri(), GufeKey(""), StructureSeries
    )
    pipeline = Pipeline.from_config_map(load_toml(TOML_CONFIG), allow_partial=True, lower=Stage.NORMALIZER)
    result: DataRef = run_prefect_ray(pipeline, data_ref)
