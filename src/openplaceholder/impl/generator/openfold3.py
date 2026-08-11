import base64
import json
import logging
import random
import re
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from openplaceholder.core.generation.generator import (
    StructureGenerator,
    StructureGeneratorArtifact,
    StructureGeneratorConfigBase,
)
from openplaceholder.core.structure import (
    Structure,
    StructureFormat,
    UnsupportedFormatError,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenFold3GeneratorConfig(StructureGeneratorConfigBase):
    sequence: str
    ligands: dict[str, str]
    n_diffusion_samples: int
    generator_directory: str | Path
    generate_n_seeds: int | None = None
    run_openfold_path: str | Path | None = None
    seeds: list[int] | None = None
    clean_up: bool = True
    # forces aggressive CPU offloading regardless of structure size; only
    # needed for very large targets that don't fit in GPU memory otherwise.
    low_mem: bool = False


class OpenFold3Generator(StructureGenerator):

    _config: OpenFold3GeneratorConfig

    def _setup(self) -> None:
        pass

    def _run(self) -> list[StructureGeneratorArtifact]:

        logger.info("preparing input files for OpenFold3 run")
        self._prepare_openfold_inputs()

        # TODO ensure that the checkpoint has been
        # downloaded. Optionally download if not found?
        if self._config.run_openfold_path:
            logger.debug("running OpenFold3 as a subprocess")
            self._run_openfold_subprocess()
        else:
            logger.debug("running OpenFold3 in-process")
            self._run_openfold_in_process()

        logger.info("gathering generated structures")
        generator_artifacts = self._package_outputs()

        if self._config.clean_up:
            logger.info("cleaning up OpenFold3 output directory")
            self._clean_up()

        return generator_artifacts

    def _prepare_openfold_inputs(self) -> None:
        gen_dir = Path(self._config.generator_directory)
        logger.debug("creating generator directory %s", gen_dir)
        gen_dir.mkdir(exist_ok=True, parents=True)

        query_json_path = gen_dir / "queries.json"
        logger.debug("creating %s", query_json_path)
        query_map: dict[Any, Any] = self._query_map()
        query_json_path.write_text(json.dumps(query_map, indent=4))

        runner_path = gen_dir / "runner.yml"
        logger.debug("creating %s", runner_path)
        runner_yaml_content: str = self._runner_yaml()
        runner_path.write_text(runner_yaml_content)

    def _query_map(self) -> dict[Any, Any]:

        queries: dict[str, Any] = {"queries": {}}
        inner_queries = queries["queries"]

        for lig_name, lig_smiles in self._config.ligands.items():
            query_name = lig_name
            inner_queries[query_name] = {
                "chains": [
                    dict(molecule_type="protein", chain_ids="A", sequence=self._config.sequence),
                    dict(molecule_type="ligand", chain_ids="Z", smiles=lig_smiles),
                ]
            }
        return queries

    def _runner_yaml(self) -> str:
        presets = "    - predict\n"
        if self._config.low_mem:
            presets += "    - low_mem\n"

        content = f"""
msa_computation_settings:
  msa_output_directory: ./msas/
  cleanup_msa_dir: False
  save_mappings: True
  msa_file_format: a3m

template_preprocessor_settings:
  output_directory: ./msas/

model_update:
  presets:
{presets}  custom:
    settings:
      memory:
        eval:
          use_deepspeed_evo_attention: false

output_writer_settings:
  structure_format: pdb

experiment_settings:
  seeds:
"""
        for seed in self._get_seeds():
            content += f"    - {seed}\n"
        return content

    def _get_seeds(self) -> list[int]:
        if self._config.seeds:
            return self._config.seeds
        elif n_seeds := self._config.generate_n_seeds:
            return [random.randint(0, 2**32 - 1) for _ in range(n_seeds)]
        else:
            # This should be caught during configuration creation
            raise RuntimeError("Seed determination failed")

    def _run_openfold_subprocess(self) -> None:
        import subprocess

        cmd = self._build_subprocess_command()
        # OpenFold3's CLI is chatty ("Seed set to ...", progress bars). Stream it
        # into our logger at debug level rather than letting it write straight to
        # the terminal, keeping the tail around to report on a failed run.
        tail: deque[str] = deque(maxlen=20)
        with subprocess.Popen(
            cmd,
            cwd=self._config.generator_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                logger.debug("openfold3: %s", line)
                tail.append(line)
            returncode = proc.wait()

        # TODO decide whether a failed run should raise here
        if returncode != 0:
            logger.error("openfold3 exited with %d; last output:\n%s", returncode, "\n".join(tail))

    def _run_openfold_in_process(self) -> None:
        # avoid implementing this for now: it would mean calling OpenFold3's
        # internal Python API directly (InferenceExperimentRunner etc.)
        # rather than its stable CLI entry point, opening us up to chasing a
        # potentially unstable API across releases.
        raise NotImplementedError

    def _failure_context(self) -> str:
        """Gather OpenFold3's own report of what went wrong, if it left one.

        OpenFold3 exits 0 even when every query fails, so a missing output
        directory is the first sign anything went wrong. Its stdout goes to our
        debug log, which leaves the summary and the per-rank error log as the
        only surviving explanation.
        """
        output_dir = Path(self._config.generator_directory) / "output"

        context = ""
        if (summary := output_dir / "summary.txt").exists():
            context += f"\n\n{summary}:\n{summary.read_text().strip()}"
        for error_log in sorted(output_dir.glob("logs/*err*.log")):
            tail = error_log.read_text().splitlines()[-20:]
            context += f"\n\n{error_log} (last {len(tail)} lines):\n" + "\n".join(tail)
        return context

    def _package_outputs(self) -> list[StructureGeneratorArtifact]:

        artifacts = []

        output_dir = Path(self._config.generator_directory) / "output"
        queries = self._query_map()["queries"]

        for query_name, query in queries.items():
            structures = []
            query_output = output_dir / query_name
            if not (query_output := output_dir / query_name).exists():
                raise RuntimeError(f"Expected output for ligand in {query_output}{self._failure_context()}")
            ligand_smiles = None
            for chain in query["chains"]:
                if ligand_smiles := chain.get("smiles"):
                    continue
            if not ligand_smiles:
                raise RuntimeError("Could not find smiles string for ligand")

            for output_file in query_output.rglob("*"):
                try:
                    structure_format = StructureFormat.from_suffix(output_file.suffix)
                except UnsupportedFormatError:
                    continue

                raw = output_file.read_bytes()
                if structure_format == StructureFormat.MMCIF:
                    raw = self._rename_ligand_residue(raw, query_name)

                structure_params = dict(
                    sequence=self._config.sequence,
                    ligand_smiles=ligand_smiles,
                    ligand_name=query_name,
                    structure_format=structure_format,
                    structure_data=base64.b64encode(raw).decode(),
                )

                structures.append(Structure(**structure_params))
            artifact = StructureGeneratorArtifact.from_structures(structures)
            artifacts.append(artifact)
        return artifacts

    @staticmethod
    def _rename_ligand_residue(raw: bytes, ligand_name: str) -> bytes:
        """Rewrite OpenFold3's generic ligand residue name to `ligand_name`.

        Each of our queries has exactly one ligand chain, so OpenFold3 always
        assigns it component ID "LIG0" (see ``smiles_to_comp_id`` in
        ``openfold3.core.data.primitives.structure.query``, which enumerates
        per-query unique ligand SMILES starting at 0). Structure.ligand_name
        is otherwise relied on as the queryable residue name (e.g.
        ``select_atoms(f"resname {ligand_name}")`` in validators), so it must
        match what's actually embedded in the structure data.
        """
        return re.sub(rb"\bLIG0\b", ligand_name.encode(), raw)

    def _build_subprocess_command(self) -> list[str]:
        exe = cast(str, self._config.run_openfold_path)
        cmd = [
            exe,
            "predict",
            "--query-json",
            "./queries.json",
            "--output-dir",
            "./output/",
            "--runner-yaml",
            "./runner.yml",
            "--num-diffusion-samples",
            str(self._config.n_diffusion_samples),
        ]
        return cmd

    def _clean_up(self) -> None:
        generator_dir = Path(self._config.generator_directory)
        shutil.rmtree(generator_dir)

    def _validate_inputs(self) -> None: ...
