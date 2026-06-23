from dataclasses import dataclass
from pathlib import Path

from openplaceholder.core.generation.generator import (
    StructureGenerator,
    StructureGeneratorArtifact,
    StructureGeneratorConfigBase,
)
from openplaceholder.impl.generator.archiver import (
    JSONArchiver,
    JSONArchiverConfig,
)


@dataclass(frozen=True, eq=True)
class JSONGeneratorConfig(StructureGeneratorConfigBase):
    path: str


class JSONGenerator(StructureGenerator):

    _config: JSONGeneratorConfig

    def _setup(self) -> None:
        self._archiver: JSONArchiver = JSONArchiver(JSONArchiverConfig(path=self._config.path))
        self.validate_inputs()

    def _run(self) -> list[StructureGeneratorArtifact]:
        return self._archiver.read()

    def _validate_inputs(self) -> None:
        if not Path(self._config.path).exists():
            raise FileNotFoundError("JSON archive does not exist")
