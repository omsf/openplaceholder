"""Command line interface for openplaceholder."""

import logging
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import click
from gufe import AlchemicalNetwork
from gufe.tokenization import GufeTokenizable

from openplaceholder.core.diagnostics import alchemicalnetwork_to_ligands_sdf
from openplaceholder.core.loader import load_toml
from openplaceholder.core.pipeline import Pipeline, Stage
from openplaceholder.core.runner import run_serial

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
    type=click.Path(dir_okay=False, path_type=Path, exists=True),
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

    match begin, end:
        case None, None:
            raise SystemExit("At least one of --begin or --end is required.")
        case str() as b, None:
            first = Stage.__members__[b.upper()]
            last = Stage.MAPPER
        case None, str() as e:
            first = Stage.GENERATOR
            last = Stage.__members__[e.upper()]
        case str() as b, str() as e:
            first = Stage.__members__[b.upper()]
            last = Stage.__members__[e.upper()]
            if first > last:
                raise SystemExit(f"'{b.lower()}' is performed after '{e.lower()}'")

    if first is Stage.GENERATOR and input is not None:
        raise SystemExit("If the beginning stage is a generator, no input should be provided.")

    if first is not Stage.GENERATOR and input is None:
        raise SystemExit(f"{first.name} requires an input (-i, --input)")

    # initial data to be used as module input
    data = None
    if input is not None:
        try:
            data = GufeTokenizable.from_json(content=input.read_text())
        except JSONDecodeError:
            raise SystemExit(f"'{input}' unable to be parsed as JSON.")

    config_map = load_toml(config)

    pipeline = Pipeline.from_config_map(config_map)
    result: GufeTokenizable = run_serial(pipeline, data)
    result.to_json(output)


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
