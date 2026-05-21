import base64
import dataclasses
import random

from openplaceholder.core.structure.structure import Structure, StructureFormat

DEFAULT_SEQUENCE = "MGSPASDPTVFHKRYLKKIRDLGEGHFGKVSLYCYDPTNDGTGEMVAVKALKADAGPQHRSGWKQEIDILRTLYHEHIIKYKGCCEDAGAASLQLVMEYVPLGSLRDYLPRHSIGLAQLLLFAQQICEGMAYLHAQHYIHRNLAARNVLLDNDRLVKIGDFGLAKAVPEGHEYYRVREDGDSPVFWYAPECLKEYKFYYASDVWSFGVTLYELLTHCDSSQSPPTKFLELIGIAQGQMTVLRLTELLERGERLPRPDKCPAEVYHLMKNCWETEASFRPTFENLIPILKTVHEKYRHHHHHH"
DEFAULT_SMILES = "C1=CC=CC=C1"
DEFAULT_LIGAND_NAME = "benzene"
DEFAULT_FORMAT = StructureFormat.MMCIF


def make_structures(
    n_structures: int = 25,
    sequence: str = DEFAULT_SEQUENCE,
    ligand_smiles: str = DEFAULT_SMILES,
    ligand_name: str = DEFAULT_LIGAND_NAME,
    structure_format: str = DEFAULT_FORMAT,
) -> list[Structure]:
    template = Structure(
        sequence=sequence,
        ligand_smiles=ligand_smiles,
        ligand_name=ligand_name,
        structure_format=structure_format,
        structure_data="",
    )
    return [
        dataclasses.replace(template, structure_data=base64.b64encode(random.randbytes(8)).decode())
        for _ in range(n_structures)
    ]
