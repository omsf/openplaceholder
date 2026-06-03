import os
import random
from pathlib import Path

from openplaceholder.impl.generator.openfold3 import (
    OpenFold3Generator,
    OpenFold3GeneratorConfig,
)

if __name__ == "__main__":

    num_seeds = 5
    gen_dir = Path() / "output_directory"
    clean_up = False

    if not (run_openfold_path := os.getenv("RUN_OPENFOLD_PATH")):
        raise ValueError("Missing RUN_OPENFOLD_PATH")

    # tyk2
    sequence = "TVFHKRYLKKIRDLGEGHFGKVSLYCYDPTNDGTGEMVAVKALKADCGPQHRSGWKQEIDILRTLYHEHIIKYKGCCEDQGEKSLQLVMEYVPLGSLRDYLPRHSIGLAQLLLFAQQICEGMAYLHAQHYIHRDLAARNVLLDNDRLVKIGDFGLAKAVPEGHEYYRVREDGDSPVFWYAPECLKEYKFYYASDVWSFGVTLYELLTHCDSSQSPPTKFLELIGIAQGQMTVLRLTELLERGERLPRPDKCPCEVYHLMKNCWETEASFRPTFENLIPILKTVHEKYQ"
    ligands = {
        "lig_ejm_31": "CC(=O)Nc1cc(NC(=O)c2c(Cl)cccc2Cl)ccn1",
        "lig_ejm_42": "CCC(=O)Nc1cc(NC(=O)c2c(Cl)cccc2Cl)ccn1",
        "lig_ejm_43": "CC(C)C(=O)Nc1cc(NC(=O)c2c(Cl)cccc2Cl)ccn1",
        "lig_ejm_46": "O=C(Nc1ccnc(NC(=O)C2CC2)c1)c1c(Cl)cccc1Cl",
        "lig_ejm_47": "O=C(Nc1ccnc(NC(=O)C2CCC2)c1)c1c(Cl)cccc1Cl",
        "lig_ejm_48": "O=C(Nc1ccnc(NC(=O)C2CCCC2)c1)c1c(Cl)cccc1Cl",
        "lig_ejm_50": "O=C(CO)Nc1cc(NC(=O)c2c(Cl)cccc2Cl)ccn1",
        "lig_jmc_23": "O=C(Nc1ccnc(NC(=O)[C@H]2C[C@H]2F)c1)c1c(Cl)cccc1Cl",
        "lig_jmc_27": "O=C(Nc1ccnc(NC(=O)[C@H]2C[C@H]2Cl)c1)c1c(Cl)cccc1Cl",
        "lig_jmc_28": "C[C@@H]1C[C@@H]1C(=O)Nc1cc(NC(=O)c2c(Cl)cccc2Cl)ccn1",
    }

    config = OpenFold3GeneratorConfig(
        sequence=sequence,
        ligands=ligands,
        n_diffusion_samples=5,
        generator_directory=gen_dir,
        run_openfold_path=run_openfold_path,
        generate_n_seeds=num_seeds,
        clean_up=clean_up,
    )

    generator = OpenFold3Generator(config)
    generator.run()
