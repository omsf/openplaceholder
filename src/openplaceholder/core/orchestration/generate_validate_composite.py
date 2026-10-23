"""Unification of Structure generation and validation."""

from typing import Iterable

from openplaceholder.core.generation.generator import StructureGenerator
from openplaceholder.core.runner import _run_single_validator
from openplaceholder.core.selection.validator import Validator
from openplaceholder.core.structure import StructureSet


class GeneratorValidatorError(Exception):
    """Raised when a validator filters something unexpected."""


def generate_and_validate(
    generator: StructureGenerator, validators: Iterable[Validator], max_attempts: int = 3
) -> StructureSet:
    """Generate a StructureSet and perform validation.

    In a real workflow, generation and validation should be run as a
    single step. This function generates structures, attempts to
    validate them, and repeats the process up to ``max_attempts``
    (default 3) until at least one Structure per ligand passes
    validation.

    Parameters
    ----------
    generator
        The ``StructureGenerator`` to use for ``Structure``
        generation.

    validators
        An iterable of ``Validator`` instances to pass the
        ``Structure`` objects through.

    max_attempts
        The maximum number calls to ``generator.run()`` that will be
        performed to find valid ``StructureSet``.

    Returns
    -------
    A ``StructureSet`` containing at least one ``Structure`` per
    ligand.

    Raises
    ------
    A ``GeneratorValidatorError`` is raised if, after
    ``max_attempts``, no valid ``StructureSet`` was found.
    """
    for _ in range(max_attempts):
        data = generator.run()
        ligand_names = {rs.ligand_name for rs in data.replicate_sets}
        for validator in validators:
            data = _run_single_validator(validator, data)
            new_names = {rs.ligand_name for rs in data.replicate_sets}

            # if the name sets are different, we've lost some ligands
            if ligand_names ^ new_names:
                break
        # all validators passed
        else:
            return data
    raise GeneratorValidatorError
