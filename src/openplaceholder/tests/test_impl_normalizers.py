import base64
from itertools import combinations
from typing import Callable

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from openplaceholder.core.structure import Structure, StructureSet
from openplaceholder.impl.normalizers import (
    BindingSiteAligner,
    BindingSiteAlignerConfig,
)
from openplaceholder.tests.datafiles import TYK2_LIG_PDB
from openplaceholder.tests.helpers import read_gzip_file

_EJM55_SMILES = "COC(=O)Nc1cc(NC(=O)c2c(Cl)cccc2Cl)ccn1"
# PDB coordinates carry three decimals, so a write/read round trip is exact to ~0.001 A
_PDB_PRECISION = 0.01


def _pose(pdb: bytes, seed: int) -> Structure:
    """The tyk2 complex with its own ligand pose, in its own arbitrary frame."""
    structure = Structure(
        sequence="X",
        ligand_smiles=_EJM55_SMILES,
        ligand_name=f"pose_{seed}",
        structure_format="pdb",
        structure_data=base64.b64encode(pdb).decode(),
    )
    rng = np.random.default_rng(seed)
    universe = structure.to_mda_universe().copy()

    # displace the ligand first, so each pose sits differently in its own
    # protein, then put the whole complex into an arbitrary frame
    ligand = universe.select_atoms("not protein and not water")
    ligand.positions = ligand.positions + rng.normal(scale=0.3, size=3)
    universe.atoms.positions = universe.atoms.positions @ Rotation.random(random_state=seed).as_matrix().T + rng.normal(
        scale=25.0, size=3
    )
    return structure.with_atoms(universe.atoms)


# Structure.to_mda_universe is cached by *value*, so equal structures share one
# universe. Each test therefore builds poses from its own seeds, keeping its
# structures (and their caches) independent of every other test's.
@pytest.fixture(scope="module")
def make_poses() -> Callable[[int], list[Structure]]:
    pdb = read_gzip_file(str(TYK2_LIG_PDB))

    def _make(first_seed: int) -> list[Structure]:
        return [_pose(pdb, seed) for seed in range(first_seed, first_seed + 3)]

    return _make


def _normalize(structures: list[Structure]) -> list[Structure]:
    aligner = BindingSiteAligner(config=BindingSiteAlignerConfig())
    return list(aligner.normalize([StructureSet.from_structures(structures)])[0])


def _max_ca_rmsd(structures: list[Structure]) -> float:
    positions = [s.to_mda_universe().select_atoms("name CA").positions for s in structures]
    return float(max(np.sqrt(((a - b) ** 2).sum(axis=1).mean()) for a, b in combinations(positions, 2)))


def _ligand_to_protein_distances(structure: Structure) -> np.ndarray:
    universe = structure.to_mda_universe()
    ligand = universe.select_atoms("not protein and not water").positions
    protein = universe.select_atoms("name CA").positions
    return np.asarray(np.linalg.norm(ligand[:, None] - protein[None], axis=-1))


def test_normalize_brings_complexes_into_a_common_frame(make_poses: Callable[[int], list[Structure]]) -> None:
    poses = make_poses(0)
    assert _max_ca_rmsd(poses) > 5.0, "the poses should start out in different frames"

    normalized = _normalize(poses)

    # every pose holds the same protein, so a shared frame makes them coincide
    assert _max_ca_rmsd(normalized) < _PDB_PRECISION


def test_normalize_preserves_each_cofolded_pose(make_poses: Callable[[int], list[Structure]]) -> None:
    poses = make_poses(10)
    normalized = {s.ligand_name: s for s in _normalize(poses)}

    for pose in poses:
        # a rigid move of the whole complex leaves the ligand where it was
        # predicted relative to its own protein
        assert_allclose(
            _ligand_to_protein_distances(normalized[pose.ligand_name]),
            _ligand_to_protein_distances(pose),
            atol=_PDB_PRECISION,
        )


def test_normalize_leaves_its_inputs_untouched(make_poses: Callable[[int], list[Structure]]) -> None:
    poses = make_poses(20)
    # ligand_atoms() caches a universe that alignto would otherwise move in place
    before = [s.ligand_atoms().positions.copy() for s in poses]

    _normalize(poses)

    for pose, original in zip(poses, before):
        assert_allclose(pose.ligand_atoms().positions, original)
