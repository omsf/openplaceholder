import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any, Self

_JSON_SERDE_CLASS_REGISTRY: dict[str, "type[JSONSerializable]"] = {}


class JSONSerializable(ABC):

    def __init_subclass__(cls: type[Self], **kwargs: dict[str, Any]) -> None:
        super().__init_subclass__(**kwargs)
        key = f"{cls.__module__}.{cls.__qualname__}"
        _JSON_SERDE_CLASS_REGISTRY[key] = cls

    @staticmethod
    def _dict_factory_hook(data: dict[Any, Any]) -> dict[Any, Any]:
        fields = []
        for key, value in data:
            if isinstance(value, JSONSerializable):
                value = value.to_dict()
            fields.append((key, value))

        return dict(fields)

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[Any, Any]) -> Self: ...

    @abstractmethod
    def to_dict(self) -> dict[Any, Any]: ...


class InvalidClassEncoding(Exception): ...


class OPHEncoder(json.JSONEncoder):

    def default(self, obj: Any) -> Any:
        match obj:
            case JSONSerializable():
                dct = obj.to_dict()
                return {"__oph_custom__": f"{obj.__class__.__module__}.{obj.__class__.__qualname__}", **dct}
            case _:
                return super().default(obj)


def oph_json_hook(dct: dict[Any, Any]) -> Any:
    """JSON decoder hook."""

    if (obj_type := dct.get("__oph_custom__", None)) is None:
        return dct
    if (cls := _JSON_SERDE_CLASS_REGISTRY.get(obj_type, None)) is not None:
        return cls.from_dict(dct)

    raise InvalidClassEncoding()


def to_shallow_dict(obj: JSONSerializable) -> dict[Any, Any]:
    return {key: value for key, value in obj.__dict__.items()}


def to_json(obj: JSONSerializable) -> str:
    return json.dumps(
        obj.to_dict() | {"__oph_custom__": f"{obj.__class__.__module__}.{obj.__class__.__qualname__}"}, cls=OPHEncoder
    )


def from_json(content: str) -> Any:
    obj = json.loads(content, object_hook=oph_json_hook)
    return obj
