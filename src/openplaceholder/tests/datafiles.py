from importlib import resources

__all__ = [
    "TYK2_LIG_PDB",
]

files = resources.files("openplaceholder.tests.data")

TYK2_LIG_PDB = files.joinpath("structures", "JACS8_tyk2_ejm55.pdb.gz")
