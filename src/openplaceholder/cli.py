"""Command line interface for openplaceholder."""

import logging
from enum import IntEnum, auto
from pathlib import Path
from typing import Any

import click
from gufe import AlchemicalNetwork

from openplaceholder.core.diagnostics import alchemicalnetwork_to_ligands_sdf
from openplaceholder.core.resolver import _build_plugin, load_toml


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
    "-b", "--begin", required=False, type=click.Choice([*map(lambda m: str(m).lower(), Stage.__members__.keys())])
)
@click.option("-i", "--input", required=False, type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "-e", "--end", required=False, type=click.Choice([*map(lambda m: str(m).lower(), Stage.__members__.keys())])
)
@click.option("-o", "--output", required=False, type=click.Path(dir_okay=False, path_type=Path))
@click.option("-v", "--verbose", is_flag=True, help="Emit debug logging.")
def run(config: Path, begin: str, end: str, output: Path, verbose: bool) -> None:
    """Run the pipeline through a beginning and end state.

    A beginning or end stage is one of: generator, validator, selector, transformation, or mapper.
    """
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    match (begin, end):
        case None, None:
            raise ValueError("At least one of --begin or --end is required.")
        case b, None:
            first = Stage.__members__[b.upper()]
            last = Stage.MAPPER
        case None, e:
            first = Stage.GENERATOR
            last = Stage.__members__[e.upper()]
        case b, e:
            first = Stage.__members__[b.upper()]
            last = Stage.__members__[e.upper()]
            if first > last:
                raise ValueError(f"'{b.lower()}' is performed after '{e.lower()}'")

    config_map = load_toml(config)

    def _resolve(path_parts):
        node = config_map
        for key in path_parts:
            if not isinstance(node, dict) or key not in node:
                raise ValueError(f"Missing required config key: {'.'.join(path_parts)}")
            node = node[key]
        return node

    config_plugin_map: dict[Stage, tuple[tuple[str, ...], bool]] = {
        Stage.GENERATOR: (("generation", "generator"), False),
        Stage.VALIDATOR: (("selection", "validators"), True),
        Stage.SELECTOR: (("selection", "selector"), False),
        Stage.TRANSFORMATION: (("assembly", "transformations"), True),
        Stage.MAPPER: (("assembly", "mapping"), False),
    }

    plugins = []
    for i in range(first, last + 1):
        path, expect_list = config_plugin_map[Stage(i)]
        node = _resolve(path)
        if expect_list:
            if not isinstance(node, list):
                raise ValueError(f"'config['{':'.join(path)}'] must be a TOML array of tables")
            stage_plugins = [_build_plugin(val) for val in node]
        else:
            stage_plugins = [_build_plugin(node)]
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
