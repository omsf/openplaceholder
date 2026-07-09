"""protonate_utils-backed protonators (PatWalters/protonate_utils, vendored).

Ligand protonation uses Dimorphite-DL microstates; protein protonation uses
hydride. The vendored source lives in ``_vendor``.
"""

import tempfile
from pathlib import Path

from rdkit import Chem

from openplaceholder.impl.protonation import _vendor
from openplaceholder.impl.protonation.base import LigandProtonator, ProteinProtonator


class ProtonateUtilsLigandProtonator(LigandProtonator):
    def protonate(self, mol: Chem.Mol, ph: float) -> Chem.Mol:
        protonated: Chem.Mol = _vendor.protonate_molecule(mol, ph)  # type: ignore[no-untyped-call]
        return protonated


class ProtonateUtilsProteinProtonator(ProteinProtonator):
    """Protein protonation via hydride.

    ``relax`` (hydride's geometry optimisation of the placed hydrogens) is off
    by default: hydride's compiled relaxation step is incompatible with the
    numpy 2.x stack here (an int32/long buffer mismatch). Hydrogen *placement*
    -- the chemically correct states for the target pH -- is unaffected.
    """

    def __init__(self, relax: bool = False):
        self._relax = relax

    def protonate(self, pdb: str, ph: float) -> str:
        import biotite.structure.io.pdb as pdb_io

        with tempfile.TemporaryDirectory() as tmp:
            in_pdb = Path(tmp) / "protein.pdb"
            in_pdb.write_text(pdb)
            structure = pdb_io.PDBFile.read(str(in_pdb)).get_structure(model=1)

            structure = _vendor.protonate_structure(structure, ph=ph, relax=self._relax)  # type: ignore[no-untyped-call]

            out_file = pdb_io.PDBFile()
            out_file.set_structure(structure)
            out_pdb = Path(tmp) / "protein_h.pdb"
            out_file.write(str(out_pdb))
            return out_pdb.read_text()
