import dataclasses
import importlib
import json
import tomllib
import typing
from pathlib import Path
from typing import Any, cast


def load_toml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    raw = tomllib.loads(path.read_text())
    return raw


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("JSON data does not contain a map")
    return raw


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


def _build_plugin(section: dict[str, Any]) -> Any:
    _section = section.copy()
    qualname: str = _section.pop("implementation")
    cls = load_class(qualname)
    config_type = resolve_config_type(cls)

    field_names = {f.name for f in dataclasses.fields(config_type)}
    if extra := set(_section) - field_names:
        msg = f"Unknown fields for {config_type.__name__}: {extra}"
        raise ValueError(msg)

    try:
        instance = cls(config=config_type(**_section))
    except TypeError as e:
        msg = f"Missing class initialization parameters: {e.args[0]}"
        raise ValueError(e)
    return instance
