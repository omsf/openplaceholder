import inspect
from abc import ABC, abstractmethod
from typing import Any, get_type_hints

from openplaceholder.core.configuration import ConfigBase


class Configurable(ABC):
    """An abstract class which enforces that enforces attribute annotions.

    Concrete subclasses of ``Configurable`` are required to have a
    ``_config`` attribute annotion whose type is a subclass of
    ``ConfigBase``. This provides a compile-time verification that
    downstream classes conform the a standard set of configure
    expectations.

    """

    def __init_subclass__(cls, **kwargs: dict[Any, Any]) -> None:
        super().__init_subclass__(**kwargs)

        # skip this check for abstract classes, only matters for concrete implementations
        if inspect.isabstract(cls):
            return

        hints = get_type_hints(cls)
        config_type = hints.get("_config", None)

        if config_type is None:
            raise TypeError(f"{cls.__name__} must annotate class attribute '_config'")

        if not issubclass(config_type, ConfigBase):
            raise TypeError(f"{cls.__name__}._config must be a subclass of 'ConfigBase'")


class Module(Configurable, ABC):
    """An abstract class for creating an OpenPlaceHolder module.

    A module is a ``Configurable`` class that performs calculations
    according to immutable configuration settings. The initializer
    checks that the passed configuration is compatible with the module
    at runtime, as specified by the ``_config`` attribute required by
    ``Configurable``.

    ``Module`` also defines the ``_setup`` abstract method. This
    should take the place of the initializer when implementing new
    modules.

    """

    def __init__(self, config: ConfigBase):
        self._check_config_type(config)
        self._config = config
        self._setup()

    def _check_config_type(self, config: Any) -> None:
        expected_config_type = get_type_hints(type(self)).get("_config")

        if expected_config_type is None:
            raise TypeError(f"{type(self).__name__} must annotate _config")

        if not issubclass(type(config), expected_config_type):
            raise TypeError(
                f"{type(self).__name__} expected config of type {expected_config_type.__name__}, got {type(config).__name__}"
            )

        if not issubclass(type(config), ConfigBase):
            raise TypeError

    @abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError
