import base64
import json
import random
import re
import shutil
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
)


@dataclass(frozen=True)
class OpenFold3GeneratorConfig(StructureGeneratorConfigBase):
    sequence: str
    ligands: dict[str, str]
    n_diffusion_samples: int
    generate_n_seeds: int | None
    generator_directory: str | Path
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
        gen_dir = Path(self._config.generator_directory)
        gen_dir.mkdir(exist_ok=True, parents=True)

        query_map: dict[Any, Any] = self._query_map()
        runner_yaml_content: str = self._runner_yaml()

        query_json_path = gen_dir / "queries.json"
        runner_path = gen_dir / "runner.yml"

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
        with subprocess.Popen(cmd, cwd=self._config.generator_directory) as proc:
            # TODO check output
            output = proc.wait()

    def _run_openfold_in_process(self) -> None:
        # mirrors openfold3.run_openfold's `predict` CLI command, calling the
        # same runner directly instead of shelling out to the CLI.
        from openfold3.core.config import config_utils
        from openfold3.entry_points.experiment_runner import InferenceExperimentRunner
        from openfold3.entry_points.validator import InferenceExperimentConfig
        from openfold3.projects.of3_all_atom.config.inference_query_format import (
            InferenceQuerySet,
        )

        gen_dir = Path(self._config.generator_directory)
        runner_args = config_utils.load_yaml(gen_dir / "runner.yml")
        expt_config = InferenceExperimentConfig(**runner_args)
        expt_runner = InferenceExperimentRunner(
            expt_config,
            num_diffusion_samples=self._config.n_diffusion_samples,
            output_dir=gen_dir / "output",
        )
        query_set = InferenceQuerySet.from_json(gen_dir / "queries.json")

        expt_runner.setup()
        expt_runner.run(query_set)
        expt_runner.cleanup()

    def _package_outputs(self) -> list[StructureGeneratorArtifact]:

        artifacts = []

        output_dir = Path(self._config.generator_directory) / "output"
        sequence = self._config.sequence
        queries = self._query_map()["queries"]

        for query_name, query in queries.items():
            structures = []
            query_output = output_dir / query_name
            if not (query_output := output_dir / query_name).exists():
                raise RuntimeError(f"Expected output for ligand in {query_output}")
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
