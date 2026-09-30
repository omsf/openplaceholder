from __future__ import annotations

import importlib
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, Protocol, Self
from urllib.parse import urlparse

from gufe.tokenization import GufeKey, GufeTokenizable


class DataRefSupportedSchemes(StrEnum):
    FILE = auto()


def get_gufe_data[T: GufeTokenizable](data_ref: DataRef[T]) -> T | None:
    """From a ``DataRef``, get the data it points to.

    Parameters
    ----------
    data_ref
        The ``DataRef`` holding the object location and key.

    Returns
    -------
    An instance of the ``GufeTokenizable`` the ``DataRef``
    references. If the object could not be found, ``None`` is
    returned.

    """

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
            raise AttributeError


class TokenizableWriter(Protocol):

    def write[T: GufeTokenizable](self, tokenizable: T) -> DataRef[T] | None: ...


class FSDataRefIO:

    # TODO: support automatic naming?
    def __init__(self, location: str):
        self.location = Path(location).resolve().as_uri()

    def write[T: GufeTokenizable](self, tokenizable: T) -> DataRef[T] | None:
        path = Path(urlparse(self.location).path)
        path.parent.mkdir(exist_ok=True, parents=True)
        try:
            tokenizable.to_json(path)
        except Exception:
            return None
        return DataRef(self.location, tokenizable.key, tokenizable.__class__)


class DataRef[T: GufeTokenizable](GufeTokenizable):  # type: ignore
    location: str
    object_key: GufeKey
    object_type: type[T]

    def __init__(self, location: str, object_key: GufeKey, object_type: type[T]):
        self.location = location
        self.object_key = object_key
        self.object_type = object_type

        self._validate_uri()

    def _validate_uri(self) -> None:
        from urllib.parse import urlparse

        result = urlparse(self.location)

        if result.scheme == "":
            raise ValueError("DataRef URI does not provide a scheme")

        if result.scheme not in DataRefSupportedSchemes.__members__:
            raise ValueError("DataRef provides an unsupported URI")

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
