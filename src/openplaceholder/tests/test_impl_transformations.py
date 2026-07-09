import base64
import importlib.util
import unittest
from unittest import TestCase

from openplaceholder.core.structure import Structure
from openplaceholder.impl.transformations import (
    ComplexProtonationTransformation,
    ComplexProtonationTransformationConfig,
    MaxVolumeSiteSelectionTransformation,
    MaxVolumeSiteSelectionTransformationConfig,
    ProteinPreparationTransformation,
    ProteinPreparationTransformationConfig,
)
from openplaceholder.tests.datafiles import TYK2_LIG_PDB
from openplaceholder.tests.helpers import read_gzip_file

# ligand protonation (dimorphite_dl) is absent from the mmcif-pr local env; CI
# (py312/py313/py314) has it, so the full-stack test runs there and skips locally
_HAS_DIMORPHITE = importlib.util.find_spec("dimorphite_dl") is not None
_EJM55_SMILES = "COC(=O)Nc1cc(NC(=O)c2c(Cl)cccc2Cl)ccn1"

# A ligand-only PDB whose 4 atoms form a tetrahedron of volume (scale^3)/6.
_TETRAHEDRON = """\
HETATM    1  C1  LIG Z   1     {0:7.3f} {1:7.3f} {1:7.3f}  1.00  0.00           C
HETATM    2  C2  LIG Z   1     {1:7.3f} {0:7.3f} {1:7.3f}  1.00  0.00           C
HETATM    3  C3  LIG Z   1     {1:7.3f} {1:7.3f} {0:7.3f}  1.00  0.00           C
HETATM    4  C4  LIG Z   1     {1:7.3f} {1:7.3f} {1:7.3f}  1.00  0.00           C
END
"""


def _ligand_structure(scale: float, name: str) -> Structure:
    pdb = _TETRAHEDRON.format(scale, 0.0)
    return Structure(
        sequence="X",
        ligand_smiles="C",
        ligand_name=name,
        structure_format="pdb",
        structure_data=base64.b64encode(pdb.encode()).decode(),
    )


class TestMaxVolumeSiteSelectionTransformation(TestCase):

    def test_returns_largest_ligand_volume_first_and_keeps_all(self) -> None:
        transformation = MaxVolumeSiteSelectionTransformation(MaxVolumeSiteSelectionTransformationConfig())
        small = _ligand_structure(2.0, "small")
        large = _ligand_structure(5.0, "large")

        selected = transformation.transform([small, large])

        self.assertEqual(selected[0].ligand_name, "large")
        self.assertEqual({s.ligand_name for s in selected}, {"small", "large"})

    def test_empty_input_returns_empty(self) -> None:
        transformation = MaxVolumeSiteSelectionTransformation(MaxVolumeSiteSelectionTransformationConfig())

        self.assertEqual(transformation.transform([]), [])


def _tyk2_complex() -> Structure:
    content = base64.b64encode(read_gzip_file(str(TYK2_LIG_PDB))).decode()
    return Structure(
        sequence="X",
        ligand_smiles=_EJM55_SMILES,
        ligand_name="ejm55",
        structure_format="pdb",
        structure_data=content,
    )


@unittest.skipUnless(_HAS_DIMORPHITE, "dimorphite_dl (ligand protonation) not installed")
class TestStackedTransformations(TestCase):
    """Select -> Prepare -> Protonate, stacked as the runner would apply them."""

    def test_full_stack_protonates_protein_and_ligand(self) -> None:
        structures = [_tyk2_complex()]

        structures = MaxVolumeSiteSelectionTransformation(MaxVolumeSiteSelectionTransformationConfig()).transform(
            structures
        )
        structures = ProteinPreparationTransformation(ProteinPreparationTransformationConfig()).transform(structures)
        structures = ComplexProtonationTransformation(ComplexProtonationTransformationConfig(ph=7.0)).transform(
            structures
        )

        self.assertEqual(len(structures), 1)
        universe = structures[0].to_mda_universe()
        self.assertGreater(len(universe.select_atoms("protein and element H")), 0)
        self.assertGreater(len(universe.select_atoms("not protein and element H")), 0)
