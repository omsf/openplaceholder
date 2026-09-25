from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Any, Iterator, Self

from openplaceholder.core.assembly.mapper import Mapper
from openplaceholder.core.assembly.transformation import Transformation
from openplaceholder.core.generation.generator import StructureGenerator
from openplaceholder.core.interface import Module
from openplaceholder.core.loader import _build_plugin
from openplaceholder.core.selection.normalizer import Normalizer
from openplaceholder.core.selection.selector import Selector
from openplaceholder.core.selection.validator import Validator


class Stage(IntEnum):
    """An integer enumeration of Stages, providing an explicit
    ordering to their execution.
    """

    GENERATOR = auto()
    VALIDATOR = auto()
    NORMALIZER = auto()
    SELECTOR = auto()
    TRANSFORMATION = auto()
    MAPPER = auto()


# Each stage, where to find its configuration based on keys, and
# whether there can be multiple instances of that stage inside the
# config (True for multiple, False for singular)
CONFIG_PLUGIN_MAP: dict[Stage, tuple[tuple[str, ...], bool]] = {
    Stage.GENERATOR: (("generation", "generator"), False),
    Stage.VALIDATOR: (("selection", "validators"), True),
    Stage.NORMALIZER: (("selection", "normalizers"), True),
    Stage.SELECTOR: (("selection", "selector"), False),
    Stage.TRANSFORMATION: (("assembly", "transformations"), True),
    Stage.MAPPER: (("assembly", "mapping"), False),
}


class PipelineResolutionError(Exception):
    """To be raised when a pipeline cannot be resolved from a
    configuration mapping.
    """

    ...


@dataclass(frozen=True)
class Pipeline:
    """Container of configured plugins."""

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

    def _get_plugin_type[T](self, plugin_type: type[T]) -> tuple[T, ...]:
        """Collect all plugins that match a certain type."""
        results: list[T] = []
        for plugin in self.plugins:
            if isinstance(plugin, plugin_type):
                results.append(plugin)
        return tuple(results)

    @property
    def generator(self) -> StructureGenerator | None:
        """The StructureGenerator in the pipeline."""
        if generator := self._get_plugin_type(StructureGenerator):  # type: ignore[type-abstract]
            return generator[0]
        return None

    @property
    def validators(self) -> tuple[Validator, ...] | None:
        """The Validator instances in the pipeline."""
        if validators := self._get_plugin_type(Validator):  # type: ignore[type-abstract]
            return validators
        return None

    @property
    def normalizers(self) -> tuple[Normalizer, ...] | None:
        """The Normalizer instances in the pipeline."""
        if normalizers := self._get_plugin_type(Normalizer):  # type: ignore[type-abstract]
            return normalizers
        return None

    @property
    def selector(self) -> Selector | None:
        """The Selector in the pipeline."""
        if selector := self._get_plugin_type(Selector):  # type: ignore[type-abstract]
            return selector[0]
        return None

    @property
    def transformations(self) -> tuple[Transformation, ...] | None:
        """The Transformation instances in the pipeline."""
        if transformations := self._get_plugin_type(Transformation):  # type: ignore[type-abstract]
            return transformations
        return None

    @property
    def mapper(self) -> Mapper | None:
        """The Mapper in the pipeline."""
        if mapper := self._get_plugin_type(Mapper):  # type: ignore[type-abstract]
            return mapper[0]
        return None

    def __iter__(self) -> Iterator[Module]:
        yield from self.plugins
