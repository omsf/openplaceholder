import base64
from unittest import TestCase

from openplaceholder.core.structure import Structure
from openplaceholder.impl.transformations import (
    MaxVolumeSiteTransformation,
    MaxVolumeSiteTransformationConfig,
)

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


class TestMaxVolumeSiteTransformation(TestCase):

    def test_select_returns_largest_ligand_volume_first_and_keeps_all(self) -> None:
        transformation = MaxVolumeSiteTransformation(MaxVolumeSiteTransformationConfig())
        small = _ligand_structure(2.0, "small")
        large = _ligand_structure(5.0, "large")

        selected = transformation._select([small, large])

        self.assertEqual(selected[0].ligand_name, "large")
        self.assertEqual({s.ligand_name for s in selected}, {"small", "large"})
