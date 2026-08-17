"""Command line interface for openplaceholder."""

import json
import logging
from enum import IntEnum, auto
from pathlib import Path
from typing import Any, cast

import click
from gufe import AlchemicalNetwork
from gufe.tokenization import GufeTokenizable

from openplaceholder.core.diagnostics import alchemicalnetwork_to_ligands_sdf
from openplaceholder.core.generation.generator import (
    StructureGeneratorArtifact,
)
from openplaceholder.core.resolver import _build_plugin, load_toml
from openplaceholder.core.selection.validator import Validator
from openplaceholder.core.serialization import (
    OPHEncoder,
    from_json,
)
from openplaceholder.core.structure import StructureSet


class Stage(IntEnum):
    GENERATOR = auto()
    VALIDATOR = auto()
    SELECTOR = auto()
    TRANSFORMATION = auto()
    MAPPER = auto()


STAGE_CHOICES = tuple(name.lower() for name in Stage.__members__)


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
    "-b",
    "--begin",
    required=False,
    type=click.Choice(STAGE_CHOICES),
    help="First stage in the pipeline.",
)
@click.option(
    "-i",
    "--input",
    required=False,
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Input JSON file for resuming the pipeline from a prior stage.",
)
@click.option(
    "-e",
    "--end",
    required=False,
    type=click.Choice(STAGE_CHOICES),
    help="Last stage in the pipeline.",
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output path for the pipeline result JSON.",
)
@click.option("-v", "--verbose", is_flag=True, help="Emit debug logging.")
def run(config: Path, begin: str | None, end: str | None, input: Path | None, output: Path, verbose: bool) -> None:
    """Run the pipeline through a beginning and end stage.

    \b
    Examples
    --------

    Run the full pipeline:

        openplaceholder run -c config.toml --end mapper -o network.json

    Generate and validate only:

        openplaceholder run -c config.toml --begin generator --end validator -o validated.json

    Resume from a prior JSON output:

        openplaceholder run -c config.toml -i generated_structures.json --begin validator -o network.json
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

    if first is Stage.GENERATOR and input is not None:
        raise ValueError("If the beginning stage is a generator, no input should be provided.")

    config_map = load_toml(config)

    def _resolve(path_parts: tuple[str, ...]) -> Any:
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
            stage_plugins = [(Stage(i), _build_plugin(val)) for val in node]
        else:
            stage_plugins = [(Stage(i), _build_plugin(node))]
        plugins.extend(stage_plugins)

    def _apply_validator(data: list[StructureGeneratorArtifact], validator: Validator) -> list[StructureSet]:
        new = []
        for artifact in data:
            validated_structures = validator.validate_structures(artifact.structures)
            new.append(StructureSet.from_structures(validated_structures))
        return new

    if input is not None:
        data = from_json(input.read_text())
    else:
        data = None

    for stage_type, plugin in plugins:
        match stage_type:
            case Stage.GENERATOR:
                data = plugin.run()
            case Stage.VALIDATOR:
                data = _apply_validator(cast(list[StructureGeneratorArtifact], data), plugin)
            case Stage.SELECTOR:
                data = plugin.select(data)
            case Stage.TRANSFORMATION:
                data = plugin.transform(data)
            case Stage.MAPPER:
                data = plugin.map(data)

    match data:
        case GufeTokenizable():
            output.write_text(cast(str, data.to_json()))
        case _:
            output.write_text(json.dumps(data, cls=OPHEncoder))


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
