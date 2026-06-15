import json
from typing import Any

from openplaceholder.core.abc import JSONSerializable
from openplaceholder.core.structure import Structure, StructureSet


class OPHEncoder(json.JSONEncoder):
    def default(self, obj: dict[Any, Any]) -> Any:
        match obj:
            case JSONSerializable():
                return {"__oph_custom__": obj.__class__.__qualname__, **_dct}
            case _:
                return super().default()


def load_json(data: str) -> dict[Any, Any]:
    """Helper function for reading JSON data with custom encoding."""
    return json.loads(data, object_hook=oph_json_hook)


class InvalidClassEncoding(Exception): ...


def oph_json_hook(dct):
    """JSON decoder hook."""
    if (obj_type := dct.get("__oph_custom__", None)) is None:
        return dct

    match obj_type:
        case "Structure":
            return Structure.from_json(obj_data)
        case "StructureSet":
            return StructureSet.from_json(obj_data)
        case _:
            raise InvalidClassEncoding()
