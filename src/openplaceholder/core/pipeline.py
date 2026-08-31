from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Any, Iterator, Self

from openplaceholder.core.interface import Module
from openplaceholder.core.loader import _build_plugin


class Stage(IntEnum):
    GENERATOR = auto()
    VALIDATOR = auto()
    SELECTOR = auto()
    TRANSFORMATION = auto()
    MAPPER = auto()


CONFIG_PLUGIN_MAP: dict[Stage, tuple[tuple[str, ...], bool]] = {
    Stage.GENERATOR: (("generation", "generator"), False),
    Stage.VALIDATOR: (("selection", "validators"), True),
    Stage.SELECTOR: (("selection", "selector"), False),
    Stage.TRANSFORMATION: (("assembly", "transformations"), True),
    Stage.MAPPER: (("assembly", "mapping"), False),
}


class PipelineResolutionError(Exception): ...


@dataclass(frozen=True)
class Pipeline:

    plugins: tuple[Module, ...]

    @classmethod
    def from_config_map(
        cls,
        config_data: dict[Any, Any],
        allow_partial: bool = False,
        lower: Stage = Stage.GENERATOR,
        upper: Stage = Stage.MAPPER,
    ) -> Self:
        """Construct a Pipeline instance from configuration data.

        Parameters
        ----------
        config_data
            A dictionary containing a pipeline spec. A missing table will raise
            a PipelineResolutionError if allow_partial is False.
        allow_partial
            Whether or not to allow subsequences of pipelines. The subsequence
            must have continuous stages. Otherwise a PipelineResolution is raised.
        lower
            The starting point of the pipeline.
        upper
            The end point of the pipeline.

        Raises
        ------
        PipelineResolutionError
            If missing tables are not found or plugin subsequences are not
            continuous.
        """

        def _resolve(path_parts: tuple[str, ...]) -> dict[Any, Any] | None:
            node = config_data
            for key in path_parts:
                if not isinstance(node, dict) or key not in node:
                    return None
                node = node[key]
            return node

        plugins = []
        last_added = None
        for stage in range(lower, upper + 1):
            path, expect_list = CONFIG_PLUGIN_MAP[Stage(stage)]
            if not (stage_config := _resolve(path)):
                if allow_partial:
                    continue
                raise PipelineResolutionError(f"Missing required config key: {'.'.join(path)}")
            if expect_list:
                plugins.extend([_build_plugin(val) for val in stage_config])
            else:
                plugins.append(_build_plugin(stage_config))

            if last_added and (stage - last_added) > 1:
                raise PipelineResolutionError("Provided stages are discontinuous")
            last_added = stage

        return cls(plugins=tuple(plugins))

    def __iter__(self) -> Iterator[Module]:
        yield from self.plugins
