from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class ConfigBase: ...
