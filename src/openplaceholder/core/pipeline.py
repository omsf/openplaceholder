from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class Step:
    instance: object


@dataclass(frozen=True, eq=True)
class Pipeline:
    generator: Step
    validators: list[Step]
    filters: list[Step] | None
    selector: Step
    # TODO: layered transformations? these might be order dependent so
    # need to be careful
    transformation: Step | None
    mapping: Step
