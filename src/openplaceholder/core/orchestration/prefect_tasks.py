"""Prefect task wrappers for OPH computations."""

from typing import Iterable

from gufe.network import AlchemicalNetwork
from gufe.tokenization import GufeKey
from prefect import task
from prefect.futures import PrefectFuture

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
from openplaceholder.core.simulation.simulator import SimulationResults
from openplaceholder.core.structure import StructureSeries, StructureSet
from openplaceholder.impl.simulators import OpenFESimulator


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


@task(name="tranformation-simulate", retries=1)
def simulate_transformation(
    simulator: OpenFESimulator, network: DataRef[AlchemicalNetwork], transformation_key: GufeKey
) -> DataRef[SimulationResults]:

    if (network_data := network.get_data()) is None:
        raise RuntimeError

    for t in network_data.edges:
        if t.key == transformation_key:
            transformation = t
            break
    else:
        raise ValueError(f"{t} not found in {network_data}")

    transformation_result = simulator.simulate_transformation(transformation)
    assert transformation_result is not None
    sim_results = SimulationResults((transformation_result,))

    if (ref := FSDataRefIO(f"./transform_cache/{sim_results.key}").write(sim_results)) is None:
        raise RuntimeError

    return ref


@task(name="network-simulate", retries=1)
def simulate_network(simulator: OpenFESimulator, network: DataRef[AlchemicalNetwork]) -> DataRef[SimulationResults]:

    if (initial_data := get_gufe_data(network)) is None:
        raise RuntimeError

    futures: list[PrefectFuture[DataRef[SimulationResults]]] = []
    for t in list(initial_data.edges):
        f = simulate_transformation.submit(simulator, network, t._key)
        futures.append(f)

    results: list[SimulationResults] = []
    for f in futures:
        if result := f.result().get_data():
            results.append(result)

    merged_result = SimulationResults.merge(results)

    if (ref := FSDataRefIO("sim_results.json").write(merged_result)) is None:
        raise RuntimeError

    return ref
