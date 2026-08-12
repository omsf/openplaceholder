import base64
import importlib.util

import numpy as np
import pytest

from openplaceholder.core.structure import Structure
from openplaceholder.impl.transformations import (
    ComplexProtonationTransformation,
    ComplexProtonationTransformationConfig,
    HeavyAtomAdditionTransformation,
    HeavyAtomAdditionTransformationConfig,
    MaxVolumeSiteSubstitutionTransformation,
    MaxVolumeSiteSubstitutionTransformationConfig,
    _ligand_volume,
)
from openplaceholder.tests.datafiles import TYK2_LIG_PDB
from openplaceholder.tests.helpers import read_gzip_file

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


def _tyk2_complex(name: str = "ejm55") -> Structure:
    content = base64.b64encode(read_gzip_file(str(TYK2_LIG_PDB))).decode()
    return Structure(
        sequence="X",
        ligand_smiles=_EJM55_SMILES,
        ligand_name=name,
        structure_format="pdb",
        structure_data=content,
    )


class TestLigandVolume:

    def test_volume_grows_with_ligand_size(self) -> None:
        assert _ligand_volume(_ligand_structure(5.0, "large")) > _ligand_volume(_ligand_structure(2.0, "small"))


class TestMaxVolumeSiteSubstitutionTransformation:

    def test_all_complexes_end_up_with_one_shared_protein(self) -> None:
        # two copies of the same complex, one with its coordinates shifted away:
        # after substitution both must carry the *canonical* protein (identical
        # coordinates), while keeping their own ligand identity.
        canonical = _tyk2_complex("canonical")
        shifted_universe = _tyk2_complex().to_mda_universe()
        shifted_universe.atoms.translate([50.0, 0.0, 0.0])
        shifted = Structure(
            sequence="X",
            ligand_smiles=_EJM55_SMILES,
            ligand_name="shifted",
            structure_format="pdb",
            structure_data=canonical.with_atoms(shifted_universe.atoms).structure_data,
        )

        result = MaxVolumeSiteSubstitutionTransformation(MaxVolumeSiteSubstitutionTransformationConfig()).transform(
            [canonical, shifted]
        )

        assert {s.ligand_name for s in result} == {"canonical", "shifted"}
        proteins = [s.protein_atoms().positions for s in result]
        assert proteins[0].shape == proteins[1].shape
        assert np.allclose(proteins[0], proteins[1], atol=1e-3)


class TestHeavyAtomAdditionTransformation:

    def test_fills_missing_heavy_atoms_without_adding_hydrogens(self) -> None:
        complex_ = _tyk2_complex()
        raw_heavy = len(complex_.protein_atoms().select_atoms("not element H"))

        prepared = HeavyAtomAdditionTransformation(HeavyAtomAdditionTransformationConfig()).transform([complex_])[0]
        protein = prepared.protein_atoms()

        # PDBFixer fills missing heavy atoms; adding hydrogens is ComplexProtonation's job, not this one's
        assert len(protein.select_atoms("not element H")) > raw_heavy
        assert len(protein.select_atoms("element H")) == 0


@pytest.mark.skipif(not _HAS_DIMORPHITE, reason="dimorphite_dl (ligand protonation) not installed")
class TestComplexProtonationTransformation:

    def test_adds_hydrogens_to_both_protein_and_ligand(self) -> None:
        complex_ = _tyk2_complex()

        protonated = ComplexProtonationTransformation(ComplexProtonationTransformationConfig(ph=7.0)).transform(
            [complex_]
        )[0]

        universe = protonated.to_mda_universe()
        assert len(universe.select_atoms("protein and element H")) > 0
        assert len(universe.select_atoms("not protein and element H")) > 0
