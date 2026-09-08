import logging
from dataclasses import dataclass
from pathlib import Path

from gufe.tokenization import GufeTokenizable

from openplaceholder.core.generation.generator import (
    ArtifactArchiver,
)
from openplaceholder.core.structure import StructureSet

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JSONArchiverConfig:
    path: str | Path


class JSONArchiver(ArtifactArchiver):

    _config: JSONArchiverConfig

    def __init__(self, config: JSONArchiverConfig):
        self._config = config

    def _read(self) -> StructureSet:
        path = Path(self._config.path)
        content = path.read_text()
        logger.debug("loaded achive data from %s", path)
        decoded = GufeTokenizable.from_json(content=content)
        return decoded  # type: ignore

    def _write(self, artifacts: StructureSet) -> None:
        path = Path(self._config.path)
        logger.debug("dumping json")

        _json = artifacts.to_json()
        logger.debug("writing json to %s", path)
        path.write_text(_json)

    def _archive_exists(self) -> bool:
        path = Path(self._config.path)
        return path.exists()
