import base64

from openplaceholder.core.structure import Structure, StructureSet
from openplaceholder.impl.validators import (
    ClashValidator,
    ClashValidatorConfig,
    PosebustersValidator,
    PosebustersValidatorConfig,
    StereoValidator,
    StereoValidatorConfig,
)

# a 4-atom pose whose connectivity cannot match the benzene SMILES template,
# so to_rdkit_ligand_mol raises LigandPerceptionError
_UNPERCEIVABLE_LIGAND = """\
HETATM    1  C1  LIG Z   1       0.000   0.000   0.000  1.00  0.00           C
HETATM    2  C2  LIG Z   1       1.500   0.000   0.000  1.00  0.00           C
HETATM    3  C3  LIG Z   1       0.000   1.500   0.000  1.00  0.00           C
HETATM    4  C4  LIG Z   1       0.000   0.000   1.500  1.00  0.00           C
END
"""


class TestStereoValidator:

    def test_init(self) -> None:
        config = StereoValidatorConfig()
        StereoValidator(config)


class TestClashValidator:

    def test_init(self) -> None:
        config = ClashValidatorConfig()
        ClashValidator(config)


class TestPosebustersValidator:

    def test_init(self) -> None:
        config = PosebustersValidatorConfig()
        PosebustersValidator(config)

    def test_drops_unperceivable_ligand_instead_of_raising(self) -> None:
        structure = Structure(
            sequence="",
            ligand_smiles="c1ccccc1",
            ligand_name="LIG",
            structure_format="pdb",
            structure_data=base64.b64encode(_UNPERCEIVABLE_LIGAND.encode()).decode(),
        )
        validator = PosebustersValidator(PosebustersValidatorConfig())
        assert validator.validate_structures(StructureSet([structure])).structures == []
