import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Self

from openplaceholder.core.configuration import ConfigBase
from openplaceholder.core.interface import Module
from openplaceholder.core.structure import StructureSet

logger = logging.getLogger(__name__)

_ARCHIVER_REGISTRY: dict[str, "type[ArtifactArchiver]"] = {}


class ArtifactArchiver(ABC):

    def __init_subclass__(cls: type[Self], **kwargs: dict[str, Any]) -> None:
        super().__init_subclass__(**kwargs)
        key = f"{cls.__module__}.{cls.__qualname__}"
        _ARCHIVER_REGISTRY[key] = cls
        logger.debug("registered ArtifactArchiver: %s", str(cls))

    @abstractmethod
    def _write(self, artifacts: StructureSet) -> None:
        raise NotImplementedError

    @abstractmethod
    def _read(self) -> StructureSet:
        raise NotImplementedError

    @abstractmethod
    def _archive_exists(self) -> bool:
        raise NotImplementedError

    def write(self, artifacts: StructureSet) -> None:
        logger.debug("writing artifacts with %s", self.__class__.__name__)
        return self._write(artifacts)

    def read(self) -> StructureSet:
        logger.debug("reading artifacts with %s", self.__class__.__name__)
        return self._read()

    def archive_exists(self) -> bool:
        return self._archive_exists()


@dataclass(frozen=True)
class StructureGeneratorConfigBase(ConfigBase):
    pass


class StructureGenerator(Module, ABC):

    def __init__(self, config: StructureGeneratorConfigBase) -> None:
        super().__init__(config)
        self.validate_inputs()

    @abstractmethod
    def _run(self) -> StructureSet:
        raise NotImplementedError

    @abstractmethod
    def _validate_inputs(self) -> None:
        raise NotImplementedError

    def run(self) -> StructureSet:
        logger.info("running %s", self.__class__.__name__)
        artifacts = self._run()
        return artifacts

    def validate_inputs(self) -> None:
        logger.info("validating %s inputs", self.__class__.__name__)
        self._validate_inputs()
