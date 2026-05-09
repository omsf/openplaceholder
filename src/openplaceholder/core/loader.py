import importlib
import typing
from typing import cast


def load_class(qualname: str) -> type:
    module_path, _, class_name = qualname.rpartition(":")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    assert type(cls) is type
    return cls


def resolve_config_type(cls: type) -> type:
    hints = typing.get_type_hints(cls.__init__)  # type: ignore[misc]
    if (config_type := hints.get("config")) is None:
        msg = f"{cls.__name__}.__init__ missing type annotated 'config' parameter"
        raise TypeError(msg)
    return cast(type, config_type)
