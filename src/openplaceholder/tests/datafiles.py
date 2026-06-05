from importlib import resources
from pathlib import Path

__all__ = [
    "TYK2_LIG_PDB",
]

_data_ref = resources.files("openplaceholder.tests.data")

TYK2_LIG_PDB = Path(_data_ref / "structures" / "JACS8_tyk2_ejm55.pdb.gz")
