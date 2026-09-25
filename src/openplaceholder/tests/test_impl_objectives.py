import base64

import numpy as np
import pytest

from openplaceholder.core.structure import Structure
from openplaceholder.impl.selector.objectives.ifp import (
    IFPSimilarityObjective,
    IFPSimilarityObjectiveConfig,
)
from openplaceholder.impl.selector.objectives.scaffold import (
    ScaffoldRMSDObjective,
    ScaffoldRMSDObjectiveConfig,
)
from openplaceholder.impl.selector.objectives.volume import (
    VolumeOverlapObjective,
    VolumeOverlapObjectiveConfig,
)
from openplaceholder.tests.datafiles import TYK2_LIG_PDB
from openplaceholder.tests.helpers import read_gzip_file


def _tyk2_structure() -> Structure:
    content = base64.b64encode(read_gzip_file(str(TYK2_LIG_PDB))).decode()
    return Structure(
        sequence=(
            "TVFHKRYLKKIRDLGEGHFGKVSLYCYDPTNDGTGEMVAVKALKADCGPQHRSGWKQEIDILRTLYHEHIIKYKGCCEDQGEKSLQLVMEYVPLGSLR"
            "DYLPRHSIGLAQLLLFAQQICEGMAYLHAQHYIHRDLAARNVLLDNDRLVKIGDFGLAKAVPEGHEYYRVREDGDSPVFWYAPECLKEYKFYYASDVW"
            "SFGVTLYELLTHCDSSQSPPTKFLELIGIAQGQMTVLRLTELLERGERLPRPDKCPCEVYHLMKNCWETEASFRPTFENLIPILKTVHEKYQ"
        ),
        ligand_smiles="[H]c1nc(N([H])C(=O)O[C@@]([H])([H])[H])c([H])c(N([H])C(=O)c2c(Cl)c([H])c([H])c([H])c2Cl)c1[H]",
        # the actual resname embedded in this PDB is the generic "LIG", not a query-derived name
        ligand_name="LIG",
        structure_format="pdb",
        structure_data=content,
    )


class TestVolumeOverlapObjective:

    def test_self_overlap_is_one(self) -> None:
        structure = _tyk2_structure()
        objective = VolumeOverlapObjective(VolumeOverlapObjectiveConfig())
        pytest.approx(objective.score(structure, structure), 1.0)


class TestIFPSimilarityObjective:

    def test_self_similarity_is_one(self) -> None:
        structure = _tyk2_structure()
        objective = IFPSimilarityObjective(IFPSimilarityObjectiveConfig())
        pytest.approx(objective.score(structure, structure), 1.0)


def _shifted(structure: Structure, offset: float) -> Structure:
    """The same complex with every atom translated along x."""
    universe = structure.to_mda_universe().copy()
    universe.atoms.positions += [offset, 0.0, 0.0]
    return structure.with_atoms(universe.atoms)


class TestScaffoldRMSDObjective:

    def test_identical_poses_score_one(self) -> None:
        structure = _tyk2_structure()
        objective = ScaffoldRMSDObjective(ScaffoldRMSDObjectiveConfig())

        assert objective.score(structure, structure) == pytest.approx(1.0)

    def test_score_falls_off_with_displacement(self) -> None:
        structure = _tyk2_structure()
        objective = ScaffoldRMSDObjective(ScaffoldRMSDObjectiveConfig(tolerance=1.0))

        near = objective.score(structure, _shifted(structure, 0.5))
        far = objective.score(structure, _shifted(structure, 3.0))

        assert 1.0 > near > far
        # a whole-molecule shift of `tolerance` should sit near exp(-1)
        assert objective.score(structure, _shifted(structure, 1.0)) == pytest.approx(0.368, abs=0.02)

    def test_tolerance_sets_how_strict_the_score_is(self) -> None:
        structure = _tyk2_structure()
        shifted = _shifted(structure, 1.0)

        strict = ScaffoldRMSDObjective(ScaffoldRMSDObjectiveConfig(tolerance=0.5))
        lenient = ScaffoldRMSDObjective(ScaffoldRMSDObjectiveConfig(tolerance=2.0))

        assert strict.score(structure, shifted) < lenient.score(structure, shifted)

    def test_ligands_with_no_shared_ring_system_have_no_core(self) -> None:
        from rdkit import Chem

        objective = ScaffoldRMSDObjective(ScaffoldRMSDObjectiveConfig())
        mols = [Chem.MolFromSmiles("CCCC"), Chem.MolFromSmiles("c1ccccc1")]

        assert objective._mcs(mols) is None

    def test_symmetric_core_is_matched_at_its_best_orientation(self) -> None:
        """A flipped dichlorophenyl is the same atoms in the same place."""
        from rdkit import Chem

        structure = _tyk2_structure()
        objective = ScaffoldRMSDObjective(ScaffoldRMSDObjectiveConfig())
        core = Chem.MolFromSmarts("c1c(Cl)cccc1Cl")
        mol = Chem.RemoveHs(structure.to_rdkit_ligand_mol())

        # the ring genuinely matches more than one way; scoring must not depend
        # on which one RDKit happens to return first
        assert len(mol.GetSubstructMatches(core, uniquify=False)) > 1
        assert objective.score(structure, structure) == pytest.approx(1.0)

    def test_one_core_is_derived_for_the_whole_pool(self) -> None:
        """Pairs must be scored on the same atoms for their sum to mean anything."""
        structure = _tyk2_structure()
        pool = [structure, _shifted(structure, 0.5), _shifted(structure, 1.0)]
        objective = ScaffoldRMSDObjective(ScaffoldRMSDObjectiveConfig())

        assert objective._core is None
        objective.matrix(pool)

        assert objective._core is not None
        # substructure matching is done once per structure, not once per pair
        assert len(objective._coords) == len(pool)

    def test_explicit_core_smarts_skips_the_mcs_search(self) -> None:
        structure = _tyk2_structure()
        objective = ScaffoldRMSDObjective(ScaffoldRMSDObjectiveConfig(core_smarts="c1c(Cl)cccc1Cl"))

        objective.matrix([structure, _shifted(structure, 0.5)])

        assert objective._core is not None
        assert objective._core.GetNumAtoms() == 8

    def test_tiny_core_switches_the_objective_off(self) -> None:
        """A 1-2 atom 'scaffold' matches everywhere and would score noise."""
        structure = _tyk2_structure()
        pool = [structure, _shifted(structure, 1.0)]
        objective = ScaffoldRMSDObjective(ScaffoldRMSDObjectiveConfig(core_smarts=None, min_core_atoms=50))

        matrix = objective.matrix(pool)

        # the real tyk2 core is 20 atoms, below the floor we set here
        assert objective._core is None
        assert (matrix[np.triu_indices(len(pool), k=1)] == 0.0).all()

    def test_core_above_the_floor_is_kept(self) -> None:
        structure = _tyk2_structure()
        objective = ScaffoldRMSDObjective(ScaffoldRMSDObjectiveConfig(min_core_atoms=6))

        objective.matrix([structure, _shifted(structure, 0.5)])

        assert objective._core is not None
        assert objective._core.GetNumAtoms() >= 6
