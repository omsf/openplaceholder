"""Command line interface for openplaceholder."""

import logging
from pathlib import Path
from typing import Any

import click
from gufe import AlchemicalNetwork

from openplaceholder.core.diagnostics import alchemicalnetwork_to_ligands_sdf
from openplaceholder.core.pipeline import Pipeline
from openplaceholder.core.resolver import resolve_pipeline
from openplaceholder.core.structure import StructureSet

logger = logging.getLogger(__name__)

# in pipeline order; running one stage runs every stage up to and including it
STAGES = ["generator", "validator", "selector", "transformer", "mapper"]


def _run_until(pipeline: Pipeline, stage: str) -> AlchemicalNetwork | None:
    """Run the pipeline, stopping once ``stage`` is done.

    Returns the network if the mapper ran, otherwise ``None``.
    """
    artifacts = pipeline.generator.run()
    click.echo(f"generated {len(artifacts)} artifacts")
    if stage == "generator":
        return None

    structure_sets = []
    for artifact in artifacts:
        structures = artifact.structures
        for validator in pipeline.validators:
            structures = validator.validate_structures(structures)
        if not structures:
            logger.warning("no structures for ligand %s passed validation, dropping", artifact.ligand_name)
            continue
        structure_sets.append(StructureSet.from_structures(structures))
    click.echo(f"validated {len(structure_sets)} structure sets")
    if stage == "validator":
        return None

    # select optimizes jointly across all ligands' candidate sets, so it is
    # called once on the full collection rather than per-ligand
    structures = pipeline.selector.select(structure_sets)
    click.echo(f"selected {len(structures)} structures")
    if stage == "selector":
        return None

    for transformation in pipeline.transformations:
        structures = transformation.transform(structures)
    click.echo(f"transformed {len(structures)} structures")
    if stage == "transformer":
        return None

    return pipeline.mapping.map(structures)


@click.group()
def cli() -> None:
    """OpenPlaceHolder: co-folding to alchemical inputs."""


@cli.command(short_help="Run the pipeline up to a stage.")
# the generated {a|b|c} metavar is long enough that click breaks the usage line
# mid-word, so name it and list the choices in the help text instead
@click.argument("stage", metavar="STAGE", type=click.Choice([*STAGES, "all"]))
@click.option(
    "-c",
    "--config",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Pipeline configuration TOML.",
)
@click.option(
    "-o",
    "--output",
    default="network.json",
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Where to write the AlchemicalNetwork, if the mapper runs.",
)
@click.option("-v", "--verbose", is_flag=True, help="Emit debug logging.")
def run(stage: str, config: Path, output: Path, verbose: bool) -> None:
    """Run the pipeline up to and including STAGE.

    STAGE is one of: generator, validator, selector, transformer, mapper, or
    'all' to run the pipeline end to end.
    """
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    pipeline = resolve_pipeline(config)
    network = _run_until(pipeline, "mapper" if stage == "all" else stage)

    if network is not None:
        click.echo(f"writing {output}")
        network.to_json(output)


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
