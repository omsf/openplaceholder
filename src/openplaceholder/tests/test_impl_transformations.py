import base64
import importlib.util

import numpy as np
import pytest

from openplaceholder.core.structure import Structure, StructureSeries
from openplaceholder.impl.transformations import (
    ComplexProtonationTransformation,
    ComplexProtonationTransformationConfig,
    ComplexSmoketestError,
    ComplexSmoketestTransformation,
    ComplexSmoketestTransformationConfig,
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
            StructureSeries([canonical, shifted])
        )

        assert {s.ligand_name for s in result.iter_series()} == {"canonical", "shifted"}
        proteins = [s.protein_atoms().positions for s in result.iter_series()]
        assert proteins[0].shape == proteins[1].shape
        assert np.allclose(proteins[0], proteins[1], atol=1e-3)


class TestHeavyAtomAdditionTransformation:

    def test_fills_missing_heavy_atoms_without_adding_hydrogens(self) -> None:
        complex_ = _tyk2_complex()
        raw_heavy = len(complex_.protein_atoms().select_atoms("not element H"))

        prepared = (
            HeavyAtomAdditionTransformation(HeavyAtomAdditionTransformationConfig())
            .transform(StructureSeries([complex_]))
            .series[0]
        )
        protein = prepared.protein_atoms()

        # PDBFixer fills missing heavy atoms; adding hydrogens is ComplexProtonation's job, not this one's
        assert len(protein.select_atoms("not element H")) > raw_heavy
        assert len(protein.select_atoms("element H")) == 0


@pytest.mark.skipif(not _HAS_DIMORPHITE, reason="dimorphite_dl (ligand protonation) not installed")
class TestComplexProtonationTransformation:

    def test_adds_hydrogens_to_both_protein_and_ligand(self) -> None:
        complex_ = _tyk2_complex()

        protonated = (
            ComplexProtonationTransformation(ComplexProtonationTransformationConfig(ph=7.0))
            .transform(StructureSeries([complex_]))
            .series[0]
        )

        universe = protonated.to_mda_universe()
        assert len(universe.select_atoms("protein and element H")) > 0
        assert len(universe.select_atoms("not protein and element H")) > 0


@pytest.fixture(scope="module")
def prepared_complex() -> Structure:
    """A tyk2 complex taken through the transformations the smoketest expects to run after."""
    series = StructureSeries([_tyk2_complex()])
    series = HeavyAtomAdditionTransformation(HeavyAtomAdditionTransformationConfig()).transform(series)
    series = ComplexProtonationTransformation(ComplexProtonationTransformationConfig(ph=7.0)).transform(series)
    return series.series[0]


def _smoketest(*structures: Structure, drop_failures: bool = False) -> StructureSeries:
    return ComplexSmoketestTransformation(ComplexSmoketestTransformationConfig(drop_failures=drop_failures)).transform(
        StructureSeries(list(structures))
    )


def _clashing(structure: Structure) -> Structure:
    """The same complex with its ligand buried in the protein."""
    # a fresh Structure: to_mda_universe is cached, so translating the
    # original's own universe would leak the clash into the other tests
    clashing: Structure = structure.copy_with_replacements(ligand_name="clashing")
    universe = clashing.to_mda_universe()
    universe.select_atoms("not protein").translate([1.5, 0.0, 0.0])
    return clashing.with_atoms(universe.atoms)


@pytest.mark.skipif(not _HAS_DIMORPHITE, reason="dimorphite_dl (ligand protonation) not installed")
class TestComplexSmoketestTransformation:

    def test_prepared_complex_passes(self, prepared_complex: Structure) -> None:
        assert [s.ligand_name for s in _smoketest(prepared_complex).iter_series()] == ["ejm55"]

    def test_unprotonated_complex_raises(self) -> None:
        # the smoketest runs last for a reason: a bare co-folded complex has
        # neither ligand nor protein hydrogens, so nothing can be parameterised
        with pytest.raises(ComplexSmoketestError, match="no explicit hydrogens"):
            _smoketest(_tyk2_complex())

    def test_clashing_pose_raises(self, prepared_complex: Structure) -> None:
        with pytest.raises(ComplexSmoketestError, match="on top of each other"):
            _smoketest(_clashing(prepared_complex))

    def test_one_failure_takes_the_whole_series_down(self, prepared_complex: Structure) -> None:
        with pytest.raises(ComplexSmoketestError, match="1 of 2 structures failed"):
            _smoketest(prepared_complex, _clashing(prepared_complex))

    def test_drop_failures_keeps_the_structures_that_passed(self, prepared_complex: Structure) -> None:
        result = _smoketest(prepared_complex, _clashing(prepared_complex), drop_failures=True)
        assert [s.ligand_name for s in result.iter_series()] == ["ejm55"]

    def test_drop_failures_still_raises_when_nothing_passes(self, prepared_complex: Structure) -> None:
        with pytest.raises(ComplexSmoketestError, match="no structures survived"):
            _smoketest(_clashing(prepared_complex), drop_failures=True)
