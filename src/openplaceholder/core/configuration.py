from dataclasses import asdict, dataclass
from typing import Any, Self

from openplaceholder.core.serialization import JSONSerializable


@dataclass(frozen=True)
class ConfigBase(JSONSerializable):

    def to_dict(self) -> dict[Any, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[Any, Any]) -> Self:
        return cls(**data)
