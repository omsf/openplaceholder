"""Prefect task wrappers for OPH computations."""

from typing import Iterable

from gufe.network import AlchemicalNetwork
from prefect import task

from openplaceholder.core.assembly.mapper import Mapper
from openplaceholder.core.assembly.transformation import Transformation
from openplaceholder.core.generation.generator import StructureGenerator
from openplaceholder.core.orchestration.generate_validate_composite import (
    generate_and_validate,
)
from openplaceholder.core.selection.normalizer import Normalizer
from openplaceholder.core.selection.selector import Selector
from openplaceholder.core.selection.validator import Validator
from openplaceholder.core.structure import StructureSeries, StructureSet


@task(name="generate", retries=1)
def generate_task(generator: StructureGenerator) -> StructureSet:
    return generator.run()


@task(name="generate-validate", retries=1)
def generate_validate_task(generator: StructureGenerator, validators: Iterable[Validator]) -> StructureSet:
    return generate_and_validate(generator, validators)


@task(name="validate", retries=1)
def validate_structures(validators: Iterable[Validator], structures: StructureSet) -> StructureSet:
    results = structures
    for validator in validators:
        results = validator.validate_structures(results)
    return results


@task(name="structure-normalization", retries=1)
def normalize_structures(normalizers: Iterable[Normalizer], structures: StructureSet) -> StructureSet:
    results = structures
    for normalizer in normalizers:
        results = normalizer.normalize(results)
    return results


@task(name="structure-selection", retries=1)
def select_structures(selector: Selector, structures: StructureSet) -> StructureSeries:
    return selector.select(structures)


@task(name="structure-transformation", retries=1)
def transform_structures(transformation: Transformation, structures: StructureSeries) -> StructureSeries:
    return transformation.transform(structures)


@task(name="structure-map", retries=1)
def map_structures(mapper: Mapper, structures: StructureSeries) -> AlchemicalNetwork:
    return mapper.map(structures)
