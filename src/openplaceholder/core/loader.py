import importlib
import typing
from typing import cast


def load_class(qualname: str) -> type:
    module_path, _, class_name = qualname.rpartition(":")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    assert isinstance(cls, type)
    return cls


def resolve_config_type(cls: type) -> type:
    # The specific config type is declared as a class-level '_config' annotation
    # on each concrete plugin (enforced by Configurable.__init_subclass__).
    hints = typing.get_type_hints(cls)
    if (config_type := hints.get("_config")) is None:
        msg = f"{cls.__name__} missing type annotated '_config' attribute"
        raise TypeError(msg)
    return cast(type, config_type)
