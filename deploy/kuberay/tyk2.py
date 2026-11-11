"""Run a TOML-configured pipeline end to end on the Ray cluster."""

import os
from pathlib import Path

from openplaceholder.core.loader import load_toml
from openplaceholder.core.orchestration.flow import run_prefect_ray
from openplaceholder.core.pipeline import Pipeline, Stage

CONFIG = Path(os.environ.get("OPH_CONFIG", "/opt/tyk2.toml"))
END = Stage.__members__[os.environ.get("OPH_END", "analyzer").upper()]


def main() -> None:
    config = load_toml(CONFIG)

    if shim := os.environ.get("OPH_RUN_OPENFOLD"):
        config["generation"]["generator"]["run_openfold_path"] = shim

    pipeline = Pipeline.from_config_map(config, lower=Stage.GENERATOR, upper=END)

    results_ref = run_prefect_ray(pipeline)
    results = results_ref.get_data()
    if results is None:
        raise RuntimeError("no results came back")

    if END is not Stage.ANALYZER:
        print(f"RESULT stopped after {END.name.lower()}: {type(results).__name__}")
        return

    print(f"RESULT analyzer={results.analyzer_name}")
    for lig_a, lig_b in results.ddg_pairs():
        if (estimate := results.ddg(lig_a, lig_b)) is not None:
            print(f"  ddG {lig_a} -> {lig_b}: {estimate[0]:.2f} +/- {estimate[1]:.2f}")
    for ligand in sorted(results.ligands()):
        if (estimate := results.dg(ligand)) is not None:
            print(f"  dG  {ligand}: {estimate[0]:.2f} +/- {estimate[1]:.2f}")

    statistics = results.meta.get("statistics", {})
    for key in ("discarded_ligands", "cycle_closure_error", "edge_error"):
        print(f"  {key}: {statistics.get(key)}")


if __name__ == "__main__":
    main()
