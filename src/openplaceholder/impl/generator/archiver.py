import base64
import json
from dataclasses import dataclass
from pathlib import Path

from openplaceholder.core.generation.generator import (
    ArtifactArchiver,
    StructureGeneratorArtifact,
)
from openplaceholder.core.serialization import OPHEncoder, from_json
from openplaceholder.core.structure import Structure, StructureFormat


@dataclass(frozen=True)
class DirectoryArchiverConfig:
    path: str


class DirectoryArchiver(ArtifactArchiver):

    def __init__(self, config: DirectoryArchiverConfig):
        self._config = config

    def _write(self, artifacts: list[StructureGeneratorArtifact]) -> None:
        root = Path(self._config.path)
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

    def _read(self) -> list[StructureGeneratorArtifact]:
        root = Path(self._config.path)
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

    def _archive_exists(self) -> bool:
        root = Path(self._config.path)
        return root.exists()


@dataclass(frozen=True)
class JSONArchiverConfig:
    path: str | Path


class JSONArchiver(ArtifactArchiver):

    _config: JSONArchiverConfig

    def __init__(self, config: JSONArchiverConfig):
        self._config = config

    def _read(self) -> list[StructureGeneratorArtifact]:
        path = Path(self._config.path)
        content = path.read_text()
        decoded = from_json(content)

        if not (isinstance(decoded, list) and all(isinstance(v, StructureGeneratorArtifact) for v in decoded)):
            raise ValueError(decoded)

        return decoded

    def _write(self, artifacts: list[StructureGeneratorArtifact]) -> None:
        path = Path(self._config.path)
        _json = json.dumps(artifacts, cls=OPHEncoder)
        path.write_text(_json)

    def _archive_exists(self) -> bool:
        path = Path(self._config.path)
        return path.exists()
