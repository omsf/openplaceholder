import base64
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from openplaceholder.core.generation.generator import StructureGeneratorArtifact
from openplaceholder.core.serialization import OPHEncoder, from_json
from openplaceholder.core.structure import Structure, StructureFormat


class ArtifactArchive(ABC):

    @abstractmethod
    def write(self, artifacts: list[StructureGeneratorArtifact]) -> None:
        raise NotImplementedError

    @abstractmethod
    def read(self) -> list[StructureGeneratorArtifact]:
        raise NotImplementedError


@dataclass(frozen=True, eq=True)
class DirectoryArchiveConfig:
    directory: str


class DirectoryArchive(ArtifactArchive):

    def __init__(self, config: DirectoryArchiveConfig):
        self._config = config

    def write(self, artifacts: list[StructureGeneratorArtifact]) -> None:
        root = Path(self._config.directory)
        root.mkdir(parents=True, exist_ok=True)
        for artifact in artifacts:
            artifact_dir = root / f"{artifact.ligand_name}"
            artifact_dir.mkdir(exist_ok=True)
            meta = {
                "sequence": artifact.sequence,
                "ligand_smiles": artifact.ligand_smiles,
                "ligand_name": artifact.ligand_name,
            }
            (artifact_dir / "artifact.json").write_text(json.dumps(meta))
            for i, structure in enumerate(artifact.structures):
                suffix = StructureFormat(structure.structure_format).to_suffix()
                (artifact_dir / f"pose_{i}{suffix}").write_bytes(structure.decode_structure_data())

    def read(self) -> list[StructureGeneratorArtifact]:
        root = Path(self._config.directory)
        artifacts: list[StructureGeneratorArtifact] = []
        for artifact_dir in filter(lambda d: d.is_dir(), root.iterdir()):

            if not (archive_file := artifact_dir / "artifact.json").exists():
                continue

            meta = json.loads(archive_file.read_text())

            structures: list[Structure] = []
            for file_path in artifact_dir.iterdir():
                if (suf := file_path.suffix) in StructureFormat.supported_formats():
                    raw = file_path.read_bytes()
                    structure_params = {
                        "structure_format": StructureFormat.from_suffix(suf),
                        "structure_data": base64.b64encode(raw).decode(),
                    } | meta
                    structures.append(Structure(**structure_params))
            if structures:
                artifacts.append(StructureGeneratorArtifact(structures=structures, **meta))
        return artifacts


@dataclass(frozen=True, eq=True)
class JSONArchiveConfig:
    path: str | Path


@dataclass(frozen=True, eq=True)
class JSONArchive(ArtifactArchive):

    def __init__(self, config: JSONArchiveConfig):
        self._config = config

    def read(self) -> list[StructureGeneratorArtifact]:
        path = Path(self._config.path)
        content = path.read_text()
        return from_json(content)

    def write(self, artifacts: list[StructureGeneratorArtifact]) -> None:
        path = Path(self._config.path)
        _json = json.dumps(artifacts, cls=OPHEncoder)
        path.write_text(_json)
