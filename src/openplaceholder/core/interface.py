import inspect
from abc import ABC, abstractmethod
from typing import Any, Self

from openplaceholder.core.configuration import ConfigBase


class Configurable(ABC):

    def __init_subclass__(cls, **kwargs: dict[Any, Any]) -> None:
        super().__init_subclass__(**kwargs)

        # skip this check for abstract classes, only matters for concrete implementations
        if inspect.isabstract(cls):
            return

        # A Configurable concrete subclass needs to define the _config
        if not (config_class := cls.__annotations__.get("_config", None)):
            raise TypeError(f"{cls.__name__} must annotate class attribute '_config'")

        if not issubclass(config_class, ConfigBase):
            raise TypeError(f"{cls.__name__}._config must be a subclass of 'ConfigBase'")


class Module(Configurable, ABC):

    def __init__(self, config: ConfigBase):
        self._config = config
        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError
