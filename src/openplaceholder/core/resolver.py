import dataclasses
import json
import tomllib
from pathlib import Path
from typing import Any

from openplaceholder.core.loader import load_class, resolve_config_type
from openplaceholder.core.pipeline import Pipeline


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


def resolve_pipeline(data: dict[str, Any]) -> Pipeline:
    """Construct a Pipeline instance from configuration data.

    Parameters
    ----------
    data
        A dictionary of pipeline component definitions.

    Returns
    -------
    Pipeline
    """
    generator = _build_plugin(data["generation"]["generator"])

    selection_table = data["selection"]
    validators = [_build_plugin(v) for v in selection_table.get("validators", [])]
    selector = _build_plugin(selection_table.get("selector"))

    assembly = data["assembly"]
    transformations = [_build_plugin(t) for t in assembly.get("transformations", [])]
    mapping_table = assembly["mapping"]
    mapping = _build_plugin(mapping_table)

    return Pipeline(
        generator=generator,
        validators=validators,
        selector=selector,
        transformations=transformations,
        mapping=mapping,
    )
