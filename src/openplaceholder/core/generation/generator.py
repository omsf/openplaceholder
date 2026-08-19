import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator, Self

from gufe.tokenization import GufeTokenizable

from openplaceholder.core.configuration import ConfigBase
from openplaceholder.core.interface import Module
from openplaceholder.core.structure import Structure, StructureSet

logger = logging.getLogger(__name__)


class StructureGeneratorArtifact(StructureSet):

    def __init__(self, structures: list[Structure], sequence: str, ligand_smiles: str, ligand_name: str) -> None:
        self.sequence = sequence
        self.ligand_smiles = ligand_smiles
        self.ligand_name = ligand_name
        super().__init__(structures=structures)

    def _to_dict(self) -> dict[Any, Any]:
        d = super()._to_dict()
        d["sequence"] = self.sequence
        d["ligand_smiles"] = self.ligand_smiles
        d["ligand_name"] = self.ligand_name
        return d

    @classmethod
    def _from_dict(cls, dct: dict[Any, Any]) -> Self:
        return cls(**dct)

    @classmethod
    def _defaults(cls) -> dict[Any, Any]:
        d = super()._defaults()
        d["sequence"] = ""
        d["ligand_smiles"] = ""
        d["ligand_name"] = ""
        return d

    @classmethod
    def from_structures(cls, structures: list[Structure]) -> Self:
        structures = StructureSet(structures).structures

        sequences = set()
        ligands_smiles = set()
        ligands_name = set()
        for structure in structures:
            sequences.add(structure.sequence)
            ligands_smiles.add(structure.ligand_smiles)
            ligands_name.add(structure.ligand_name)

        if len(sequences) > 1 or len(ligands_smiles) > 1 or len(ligands_name) > 1:
            raise ValueError("Structures do not represent the same complex")

        sequence = sequences.pop()
        ligand_smiles = ligands_smiles.pop()
        ligand_name = ligands_name.pop()

        return cls(structures=structures, sequence=sequence, ligand_smiles=ligand_smiles, ligand_name=ligand_name)


class ArtifactBundle(GufeTokenizable):

    def __init__(self, artifacts: list[StructureGeneratorArtifact]):
        self.artifacts = artifacts

    @classmethod
    def _defaults(cls) -> dict[Any, Any]:
        return {"artifacts": []}

    @classmethod
    def _from_dict(cls, dct: dict[Any, Any]) -> Self:
        return cls(**dct)

    def _to_dict(self) -> dict[Any, Any]:
        return {"artifacts": self.artifacts}

    def __iter__(self) -> Iterator[StructureGeneratorArtifact]:
        return iter(self.artifacts)

    def __len__(self) -> int:
        return len(self.artifacts)

    def __getitem__(self, key: int) -> StructureGeneratorArtifact:
        return self.artifacts[key]


_ARCHIVER_REGISTRY: dict[str, "type[ArtifactArchiver]"] = {}


class ArtifactArchiver(ABC):

    def __init_subclass__(cls: type[Self], **kwargs: dict[str, Any]) -> None:
        super().__init_subclass__(**kwargs)
        key = f"{cls.__module__}.{cls.__qualname__}"
        _ARCHIVER_REGISTRY[key] = cls
        logger.debug("registered ArtifactArchiver: %s", str(cls))

    @abstractmethod
    def _write(self, artifacts: ArtifactBundle) -> None:
        raise NotImplementedError

    @abstractmethod
    def _read(self) -> ArtifactBundle:
        raise NotImplementedError

    @abstractmethod
    def _archive_exists(self) -> bool:
        raise NotImplementedError

    def write(self, artifacts: ArtifactBundle) -> None:
        logger.debug("writing artifacts with %s", self.__class__.__name__)
        return self._write(artifacts)

    def read(self) -> ArtifactBundle:
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
    def _run(self) -> ArtifactBundle:
        raise NotImplementedError

    @abstractmethod
    def _validate_inputs(self) -> None:
        raise NotImplementedError

    def run(self) -> ArtifactBundle:
        logger.info("running %s", self.__class__.__name__)
        artifacts = self._run()
        return artifacts

    def validate_inputs(self) -> None:
        logger.info("validating %s inputs", self.__class__.__name__)
        self._validate_inputs()
