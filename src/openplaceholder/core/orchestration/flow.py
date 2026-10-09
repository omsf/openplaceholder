from gufe.tokenization import GufeTokenizable
from prefect import flow

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
def run_prefect(pipeline: Pipeline, initial_data: GufeTokenizable | None = None) -> GufeTokenizable:
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
        data = initial_data
    # only generate, don't validate
    elif generator is not None and validators is None:
        data = generate_task(generator)
        return data
    # pre-generated data
    elif generator is None and validators is not None:
        data = initial_data
        if not isinstance(data, StructureSet):
            raise TypeError
        data = validate_structures(validators, data)
    # generate and validate
    elif generator is not None and validators is not None:
        data = generate_validate_task(generator, validators)

    if normalizers := pipeline.normalizers:
        if not isinstance(data, StructureSet):
            raise TypeError
        data = normalize_structures(normalizers, data)

    if selector := pipeline.selector:
        if not isinstance(data, StructureSet):
            raise TypeError
        data = select_structures(selector, data)

    if transformations := pipeline.transformations:
        for transformation in transformations:
            if not isinstance(data, StructureSeries):
                raise TypeError
            data = transform_structures(transformation, data)

    if mapper := pipeline.mapper:
        if not isinstance(data, StructureSeries):
            raise TypeError
        data = map_structures(mapper, data)

    if not isinstance(data, GufeTokenizable):
        raise TypeError

    return data
