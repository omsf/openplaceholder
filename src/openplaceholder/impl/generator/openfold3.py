import base64
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from openplaceholder.core.generation.generator import (
    StructureGenerator,
    StructureGeneratorArtifact,
)
from openplaceholder.core.structure import Structure, StructureFormat


@dataclass(frozen=True, eq=True)
class OpenFold3GeneratorConfig:
    sequence: str
    ligands: dict[str, str]
    n_diffusion_samples: int
    generate_n_seeds: int | None
    output_directory: str | Path
    run_openfold_path: str | Path | None = None
    seeds: list[int] | None = None
    clean_up: bool = True


class OpenFold3Generator(StructureGenerator):

    def __init__(self, config: OpenFold3GeneratorConfig):
        self._config = config

    def run(self) -> list[StructureGeneratorArtifact]:

        self._prepare_openfold_inputs()

        # TODO ensure that the checkpoint has been
        # downloaded. Optionally download if not found?
        if self._config.run_openfold_path:
            self._run_openfold_subprocess()
        else:
            self._run_openfold_in_process()

        generator_artifacts = self._package_outputs()

        if self._config.clean_up:
            self._clean_up()

        return generator_artifacts

    def _prepare_openfold_inputs(self) -> None:
        output_dir = Path(self._config.output_directory)
        output_dir.mkdir(exist_ok=True, parents=True)

        query_map: dict[Any, Any] = self._query_map()
        runner_yaml_content: str = self._runner_yaml()

        query_json_path = output_dir / "queries.json"
        runner_path = output_dir / "runner.yml"

        query_json_path.write_text(json.dumps(query_map, indent=4))
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
        # TODO add diffusion samples number
        content = """
msa_computation_settings:
  msa_output_directory: ./msas/
  cleanup_msa_dir: False
  save_mappings: True
  msa_file_format: a3m

template_preprocessor_settings:
  output_directory: ./msas/

model_update:
  presets:
    - predict
    - low_mem
  custom:
    settings:
      memory:
        eval:
          use_deepspeed_evo_attention: false

experiment_settings:
  seeds:
"""
        for seed in self._get_seeds:
            content += f"    - {seed}\n"
        return content

    def _get_seeds(self) -> list[str]:

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
        with subprocess.Popen(cmd, cwd=self._config.output_directory) as proc:
            # TODO check output
            output = proc.wait()

    def _run_openfold_in_process(self) -> None:
        raise NotImplementedError

    def _package_outputs(self) -> list[StructureGeneratorArtifact]:

        artifacts = []

        output_dir = Path(self._config.output_directory)
        sequence = self._config.sequence
        queries = self._query_map()["queries"]

        for query_name, query in queries.items():
            structures = []
            query_output = output_dir / query_name
            ligand_smiles = None
            for chain in query["chains"]:
                if ligand_smiles := chain.get("smiles"):
                    continue
            if not ligand_smiles:
                raise RuntimeError("Could not find smiles string for ligand")

            for output_file in query_output.rglob("*"):
                try:
                    structure_format = StructureFormat.from_suffix(output_file.suffix)
                except ValueError:
                    continue

                structure_params = dict(
                    sequence=self._config.sequence,
                    ligand_smiles=ligand_smiles,
                    ligand_name=query_name,
                    structure_format=StructureFormat.from_suffix(output_file.suffix),
                    structure_data=base64.b64encode(output_file.read_bytes()).decode(),
                )

                structures.append(Structure(**structure_params))
            artifact = StructureGeneratorArtifact.from_structures(structures)
            artifacts.append(artifact)
        return artifacts

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
        ]
        return cmd

    def _clean_up(self) -> None:
        output_dir = Path(self._config.output_directory)
        shutil.rmtree(output_dir)

    def validate_input(self) -> None:
        raise NotImplementedError
