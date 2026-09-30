from prefect import flow

from openplaceholder.core.io import DataRef
from openplaceholder.core.orchestration.prefect_tasks import (
    generate_task,
    generate_validate_task,
    map_structures,
    normalize_structures,
    select_structures,
    transform_structures,
    validate_structures,
)
from openplaceholder.core.pipeline import Pipeline
from openplaceholder.core.structure import StructureSeries, StructureSet


@flow(name="openplaceholder-pipeline")
def run_prefect(pipeline: Pipeline, initial_data: DataRef | None = None) -> DataRef:  # type: ignore[type-arg]
    """Prefect flow for executing an entire OPH workflow.

    Parameters
    ----------
    pipeline
        The pipeline to be executed
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

    if generator is None and validators is None:
        if initial_data is None:
            raise ValueError("Initial data must be provided for steps after generation")
    # only generate, don't validate
    elif generator is not None and validators is None:
        data: DataRef[StructureSet] = generate_task.submit(generator).result()
        return data
    # pre-generated data, still needs validation
    elif generator is None and validators is not None:
        if initial_data is None:
            raise ValueError("Initial data must be provided for steps after generation")
        if initial_data.object_type is not StructureSet:
            raise TypeError(
                f"Initial data has incompatible type. Expected StructureSet, found {initial_data.object_type}"
            )
        data = validate_structures.submit(validators, initial_data).result()
    # generate and validate
    elif generator is not None and validators is not None:
        data = generate_validate_task.submit(generator, validators).result()

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

    return data
