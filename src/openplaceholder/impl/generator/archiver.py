import base64
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from gufe.tokenization import GufeTokenizable

from openplaceholder.core.generation.generator import (
    ArtifactArchiver,
    ArtifactBundle,
    StructureGeneratorArtifact,
)
from openplaceholder.core.structure import Structure, StructureFormat

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DirectoryArchiverConfig:
    path: str


class DirectoryArchiver(ArtifactArchiver):

    def __init__(self, config: DirectoryArchiverConfig):
        self._config = config

    def _write(self, artifacts: ArtifactBundle) -> None:
        root = Path(self._config.path)
        logger.debug("creating root archive directory: %s", root)
        root.mkdir(parents=True, exist_ok=True)
        for artifact in artifacts:
            logger.debug("achiving structures for ligand %s", artifact.ligand_name)
            artifact_dir = root / str(artifact.ligand_name)
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

    def _read(self) -> ArtifactBundle:
        root = Path(self._config.path)
        artifacts = []
        for artifact_dir in filter(lambda d: d.is_dir(), root.iterdir()):

            if not (archive_file := artifact_dir / "artifact.json").exists():
                continue

            logger.debug("loading discovered metadata from %s", archive_file)
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
        return ArtifactBundle(artifacts)

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

    def _read(self) -> ArtifactBundle:
        path = Path(self._config.path)
        content = path.read_text()
        logger.debug("loaded achive data from %s", path)
        decoded = GufeTokenizable.from_json(content=content)
        return decoded

    def _write(self, artifacts: ArtifactBundle) -> None:
        path = Path(self._config.path)
        logger.debug("dumping json")

        _json = artifacts.to_json()
        logger.debug("writing json to %s", path)
        path.write_text(_json)

    def _archive_exists(self) -> bool:
        path = Path(self._config.path)
        return path.exists()
