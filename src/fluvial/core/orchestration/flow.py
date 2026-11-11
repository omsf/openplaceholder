from gufe.network import AlchemicalNetwork
from prefect import flow
from prefect_ray import RayTaskRunner

from fluvial.core.assembly.mapper import Mapper
from fluvial.core.assembly.transformation import Transformation
from fluvial.core.io import DataRef
from fluvial.core.orchestration.prefect_tasks import (
    generate_task,
    generate_validate_task,
    map_structures,
    normalize_structures,
    select_structures,
    simulate_network,
    transform_structures,
    validate_structures,
)
from fluvial.core.pipeline import Pipeline, Stage
from fluvial.core.selection.normalizer import Normalizer
from fluvial.core.selection.selector import Selector
from fluvial.core.simulation.simulator import SimulationResults, Simulator
from fluvial.core.structure import StructureSeries, StructureSet
from fluvial.impl.simulators import OpenFESimulator

FLVRefTokenizables = (
    DataRef[StructureSet] | DataRef[StructureSeries] | DataRef[AlchemicalNetwork] | DataRef[SimulationResults]
)


@flow(name="fluvial-pipeline", task_runner=RayTaskRunner(init_kwargs={"num_cpus": 4}))
def run_prefect_ray(pipeline: Pipeline, initial_data: FLVRefTokenizables | None = None) -> FLVRefTokenizables:
    """Prefect flow for executing an entire FLV workflow using ``ray``.

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
    A TypeErnror is raised if the output type from a module's return is
    unexpected.

    """

    # if it's a generator, collect all validators if they exist
    generator = pipeline.generator
    validators = pipeline.validators

    if generator is None and validators is None and initial_data is None:
        raise ValueError("Initial data must be provided for steps after generation")

    data: FLVRefTokenizables

    match (generator, validators, initial_data):
        case (None, None, None):
            raise ValueError("Initial data must be provided for steps after generation")

        # we were only given data, pass this along to the next step
        case (None, None, d) if d is not None:
            data = d

        # generator only
        case (g, None, None) if g is not None:
            data = generate_task.submit(g).result()
            return data

        # validator only, missing the needed initial data
        case (None, _, None):
            raise ValueError

        case (g, v, None) if g is not None and v is not None:
            data = generate_validate_task.submit(g, v).result()

        case (None, v, d) if d is not None and d.object_type is StructureSet and v is not None:
            data = validate_structures.submit(v, d).result()

        # given a generator with data
        case (_, None, _):
            raise ValueError("Initial data should not be  provided for a generator")

        # anything else
        case (_, _, _):
            raise RuntimeError

    if not isinstance(data, DataRef):
        raise ValueError

    assert isinstance(data, DataRef)

    for module in pipeline.iter_range(lower=Stage.NORMALIZER):
        match (module, data):
            case (Normalizer(), d) if d.object_type is StructureSet:
                data = normalize_structures.submit([module], d).result()
            case (Selector(), d) if d.object_type is StructureSet:
                data = select_structures.submit(module, d).result()
            case (Transformation(), d) if d.object_type is StructureSeries:
                data = transform_structures.submit(module, d).result()
            case (Mapper(), d) if d.object_type is StructureSeries:
                data = map_structures.submit(module, d).result()
            case (Simulator(), d) if d.object_type is AlchemicalNetwork:
                if not isinstance(module, OpenFESimulator):
                    raise TypeError("Prefect orchestratin only supports OpenFESimulator")
                data = simulate_network.submit(module, d).result()

    return data
