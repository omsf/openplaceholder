"""Command line interface for openplaceholder."""

import logging
from enum import IntEnum, auto
from pathlib import Path
from typing import Any

import click
from gufe import AlchemicalNetwork

from openplaceholder.core.diagnostics import alchemicalnetwork_to_ligands_sdf
from openplaceholder.core.resolver import _build_plugin, load_toml

logger = logging.getLogger(__name__)


class Stage(IntEnum):
    GENERATOR = auto()
    VALIDATOR = auto()
    SELECTOR = auto()
    TRANSFORMATION = auto()
    MAPPER = auto()


@click.group()
def cli() -> None:
    """OpenPlaceHolder: co-folding to alchemical inputs."""


@cli.command(short_help="Run the pipeline fully or partially.")
@click.option(
    "-c",
    "--config",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Pipeline configuration TOML.",
)
@click.option(
    "-s",
    "--stage",
    required=False,
    type=click.Choice(["all", *map(lambda m: str(m).lower(), Stage.__members__.keys())]),
)
@click.option(
    "-b", "--begin", required=False, type=click.Choice([*map(lambda m: str(m).lower(), Stage.__members__.keys())])
)
@click.option("-i", "--input", required=False, type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "-e", "--end", required=False, type=click.Choice([*map(lambda m: str(m).lower(), Stage.__members__.keys())])
)
@click.option("-o", "--output", required=False, type=click.Path(dir_okay=False, path_type=Path))
@click.option("-v", "--verbose", is_flag=True, help="Emit debug logging.")
def run(config: Path, begin: str, end: str, stage: str, output: Path, verbose: bool) -> None:
    """Run the pipeline up to and including STAGE.

    STAGE is one of: generator, validator, selector, transformer, mapper, or
    'all' to run the pipeline end to end.
    """
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    match (begin, end, stage):
        case None, None, s:
            if s.lower() == "all":
                first = Stage.GENERATOR
                last = Stage.MAPPER
            else:
                first = last = Stage.__members__[s.upper()]
        case None, None, "all":
            first = Stage.GENERATOR
            last = Stage.MAPPER
        case b, None, None:
            first = Stage.__members__[b.upper()]
            last = Stage.MAPPER
        case None, e, None:
            first = Stage.GENERATOR
            last = Stage.__members__[e.upper()]
        case b, e, None:
            first = Stage.__members__[b.upper()]
            last = Stage.__members__[e.upper()]
            if first > last:
                raise ValueError(f"'{b.lower()}' is performed after '{e.lower()}'")
        case _:
            raise ValueError("Must supply a beginning and end, only beginning, only end, or a specific stage.")

    config_map = load_toml(config)

    plugins = []
    for i in range(first, last + 1):
        match i:
            case Stage.GENERATOR:
                stage_plugins = [_build_plugin(config_map["generation"]["generator"])]
            case Stage.VALIDATOR:
                stage_plugins = [_build_plugin(val) for val in config_map["selection"]["validators"]]
            case Stage.SELECTOR:
                stage_plugins = [_build_plugin(config_map["selection"]["selector"])]
            case Stage.TRANSFORMATION:
                stage_plugins = [_build_plugin(trans) for trans in config_map["assembly"]["transformations"]]
            case Stage.MAPPER:
                stage_plugins = [_build_plugin(config_map["assembly"]["mapping"])]
        plugins.extend(stage_plugins)


@cli.group()
def diagnostics() -> None:
    """Inspect pipeline outputs."""


@diagnostics.command("ligands-sdf", short_help="Write a network's ligands to SDF.")
@click.argument("network", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", default="-", type=click.File("w"), help="SDF destination [default: stdout].")
def ligands_sdf(network: Path, output: Any) -> None:
    """Write the ligands of an AlchemicalNetwork JSON to the SDF format."""
    alchemicalnetwork_to_ligands_sdf(AlchemicalNetwork.from_json(network), output)


if __name__ == "__main__":
    cli()
