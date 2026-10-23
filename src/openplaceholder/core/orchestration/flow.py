from gufe.network import AlchemicalNetwork
from prefect import flow
from prefect_ray import RayTaskRunner

from openplaceholder.core.io import DataRef
from openplaceholder.core.orchestration.prefect_tasks import (
    generate_task,
    generate_validate_task,
    map_structures,
    normalize_structures,
    select_structures,
    simulate_network,
    transform_structures,
    validate_structures,
)
from openplaceholder.core.pipeline import Pipeline
from openplaceholder.core.simulation.simulator import SimulationResults
from openplaceholder.core.structure import StructureSeries, StructureSet
from openplaceholder.impl.simulators import OpenFESimulator

OPHRefTokenizables = (
    DataRef[StructureSet] | DataRef[StructureSeries] | DataRef[AlchemicalNetwork] | DataRef[SimulationResults]
)


@flow(name="openplaceholder-pipeline", task_runner=RayTaskRunner(init_kwargs={"num_cpus": 4}))
def run_prefect_ray(pipeline: Pipeline, initial_data: OPHRefTokenizables | None = None) -> OPHRefTokenizables:
    """Prefect flow for executing an entire OPH workflow.

    Parameters
    ----------
    pipeline
        The pipeline to be executed.
    initial_data
        The initial data needed to run the pipeline. The type of this
        parameter depends on the first module in the pipeline.

    Returns
    -------
    The output type of the last module in the pipeline.

    Raises
    ------
    A TypeError is raised if the output type from a module's return is
    unexpected.

    """

    # if it's a generator, collect all validators if they exist
    generator = pipeline.generator
    validators = pipeline.validators

    if generator is None and validators is None and initial_data is None:
        raise ValueError("Initial data must be provided for steps after generation")

    match (generator, validators):
        # generator only, no initial data should be present
        case (g, None) if g is not None:

            if initial_data is not None:
                raise ValueError(f"Expected `None` for initial_data, got `{initial_data}`")

            data: DataRef[StructureSet] = generate_task.submit(g).result()
            return data

        case (None, v) if v is not None:

            if initial_data is None:
                raise ValueError("`initial_data` must not be `None`")

            if initial_data.object_type is not StructureSet:
                raise TypeError(
                    f"Initial data has incompatible type. Expected `StructureSet`, got {initial_data.object_type}"
                )

            data = validate_structures.submit(v, initial_data).result()
        case (g, v) if g is not None and v is not None:
            data = generate_validate_task.submit(g, v).result()

    if normalizers := pipeline.normalizers:
        if data.object_type is not StructureSet:
            raise TypeError
        data = normalize_structures.submit(normalizers, data).result()

    if selector := pipeline.selector:
        if data.object_type is not StructureSet:
            raise TypeError
        data = select_structures.submit(selector, data).result()

    if transformations := pipeline.transformations:
        for transformation in transformations:
            if data.object_type is not StructureSeries:
                raise TypeError
            data = transform_structures.submit(transformation, data).result()

    if mapper := pipeline.mapper:
        if data.object_type is not StructureSeries:
            raise TypeError
        data = map_structures.submit(mapper, data).result()

    if simulator := pipeline.simulator:
        if data.object_type is not AlchemicalNetwork:
            raise ValueError(f"{data.object_type}")
        if not isinstance(simulator, OpenFESimulator):
            raise ValueError
        data = simulate_network.submit(simulator, data).result()

    return data
