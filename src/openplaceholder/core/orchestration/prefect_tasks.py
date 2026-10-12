"""Prefect task wrappers for OPH computations."""

from typing import Iterable

from gufe.network import AlchemicalNetwork
from prefect import task

from openplaceholder.core.assembly.mapper import Mapper
from openplaceholder.core.assembly.transformation import Transformation
from openplaceholder.core.generation.generator import StructureGenerator
from openplaceholder.core.io import DataRef, FSDataRefIO, get_gufe_data
from openplaceholder.core.orchestration.generate_validate_composite import (
    generate_and_validate,
)
from openplaceholder.core.selection.normalizer import Normalizer
from openplaceholder.core.selection.selector import Selector
from openplaceholder.core.selection.validator import Validator
from openplaceholder.core.structure import StructureSeries, StructureSet


@task(name="generate", retries=1)
def generate_task(generator: StructureGenerator) -> DataRef[StructureSet]:
    data = generator.run()
    reference: DataRef[StructureSet] | None = FSDataRefIO("generator.json").write(data)
    if reference is not None:
        return reference
    raise RuntimeError


@task(name="generate-validate", retries=1)
def generate_validate_task(generator: StructureGenerator, validators: Iterable[Validator]) -> DataRef[StructureSet]:
    data = generate_and_validate(generator, validators)
    reference: DataRef[StructureSet] | None = FSDataRefIO("validated.json").write(data)
    if reference is not None:
        return reference
    raise RuntimeError


@task(name="validate", retries=1)
def validate_structures(validators: Iterable[Validator], structures: DataRef[StructureSet]) -> DataRef[StructureSet]:
    results: StructureSet | None = get_gufe_data(structures)

    if results is None:
        raise RuntimeError

    for validator in validators:
        results = validator.validate_structures(results)

    ref = FSDataRefIO("validated.json").write(results)

    if ref is None:
        raise RuntimeError

    return ref


@task(name="structure-normalization", retries=1)
def normalize_structures(normalizers: Iterable[Normalizer], structures: DataRef[StructureSet]) -> DataRef[StructureSet]:
    if (results := get_gufe_data(structures)) is None:
        raise RuntimeError

    for normalizer in normalizers:
        results = normalizer.normalize(results)

    if (ref := FSDataRefIO("normalized.json").write(results)) is None:
        raise RuntimeError

    return ref


@task(name="structure-selection", retries=1)
def select_structures(selector: Selector, structures: DataRef[StructureSet]) -> DataRef[StructureSeries]:

    if (results := get_gufe_data(structures)) is None:
        raise RuntimeError

    data = selector.select(results)

    if (ref := FSDataRefIO("selected").write(data)) is None:
        raise RuntimeError

    return ref


@task(name="structure-transformation", retries=1)
def transform_structures(
    transformation: Transformation, structures: DataRef[StructureSeries]
) -> DataRef[StructureSeries]:

    if (results := get_gufe_data(structures)) is None:
        raise RuntimeError

    data = transformation.transform(results)

    if (ref := FSDataRefIO("transformed.json").write(data)) is None:
        raise RuntimeError

    return ref


@task(name="structure-map", retries=1)
def map_structures(mapper: Mapper, structures: DataRef[StructureSeries]) -> DataRef[AlchemicalNetwork]:

    if (initial_data := get_gufe_data(structures)) is None:
        raise RuntimeError

    data = mapper.map(initial_data)

    if (ref := FSDataRefIO("alchemical_network.json").write(data)) is None:
        raise RuntimeError

    return ref
