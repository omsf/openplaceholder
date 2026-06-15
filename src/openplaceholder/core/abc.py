from abc import ABC, abstractmethod
from typing import Any, Self


class JSONSerializable(ABC):

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[Any, Any]) -> Self: ...

    @abstractmethod
    def to_dict(self) -> dict[Any, Any]: ...
