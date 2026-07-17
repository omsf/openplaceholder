import logging
import tomllib
from pathlib import Path

from openplaceholder.core.structure import StructureSet
from openplaceholder.core.resolver import _build_plugin

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("OPH-RUN-MAPPER")

TOML_CONFIG = Path(__file__).parents[1] / "config.toml"

if __name__ == "__main__":

    if not TOML_CONFIG.exists():
        raise FileNotFoundError(f"Could not find {TOML_CONFIG}")
    config = tomllib.loads(TOML_CONFIG.read_text())

    mapper = _build_plugin(config["assembly"]["mapping"])
    logger.info("Created plugin instance, %s", mapper)

    post_transformation_structures = StructureSet.from_file("post_transformation_structures.json")

    logger.info("Read post-transformation structures from disk: %d structures found", len(post_transformation_structures))

    network = mapper.map(post_transformation_structures)
    network.to_json("network.json")
