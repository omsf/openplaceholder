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
    # Concrete plugins inherit Module.__init__ rather than overriding it (see
    # Module's docstring), so the specific config type must come from the
    # class-level '_config' annotation instead of __init__'s type hints.
    hints = typing.get_type_hints(cls)
    if (config_type := hints.get("_config")) is None:
        msg = f"{cls.__name__} missing type annotated '_config' attribute"
        raise TypeError(msg)
    return cast(type, config_type)
