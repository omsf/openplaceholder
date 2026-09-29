from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Self, Protocol
from urllib.parse import urlparse

from gufe.tokenization import GufeKey, GufeTokenizable


def get_gufe_data[T: GufeTokenizable](data_ref: DataRef[T]) -> T | None:
    """From a DataRef, get the data it points to."""

    parsed = urlparse(data_ref.location)

    match parsed.scheme:
        case "file":
            data_path = Path(parsed.path)

            if not data_path.exists():
                return None

            obj = data_ref.object_type.from_json(content=data_path.read_text())

            if not isinstance(obj, data_ref.object_type):
                raise TypeError("Found unexpected type")

            return obj
        case _:
            raise TypeError


def write_disk[T: GufeTokenizable](write_location: str, tokenizable: T) -> DataRef[T] | None:
    """Write a GufeTokenizable to disk."""
    output_path = Path(write_location)
    tokenizable.to_json(output_path)
    return DataRef(f"file:{output_path.absolute()}", tokenizable.key, tokenizable.__class__)

class TokenizableReader(Protocol):

    def read[T: GufeTokenizable](self, reference: DataRef[T]) -> T | None: ...

class TokenizableWriter(Protocol):

    def write[T: GufeTokenizable](self, tokenizable: T) -> DataRef[T] | None: ...


class DataRef[T: GufeTokenizable](GufeTokenizable):  # type: ignore
    location: str
    object_key: GufeKey
    object_type: type[T]

    def __init__(self, location: str, object_key: GufeKey, object_type: type[T]):
        self.location = location
        self.object_key = object_key
        self.object_type = object_type

    def get_data(self) -> T | None:
        return get_gufe_data(self)

    def _to_dict(self) -> dict[Any, Any]:
        dct = {
            "location": self.location,
            "object_key": self.object_key,
            "object_type": [self.object_type.__module__, self.object_type.__name__],
        }
        return dct

    @classmethod
    def _defaults(cls) -> dict[Any, Any]:
        return {}

    @classmethod
    def _from_dict(cls, dct: dict[Any, Any]) -> Self:

        match dct.get("object_type"):
            case [module_name, class_name]:
                module = importlib.import_module(module_name)
                dct.update({"object_type": getattr(module, class_name)})
            case _:
                raise TypeError
        return cls(**dct)
