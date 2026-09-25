"""Scaffold objective: rewards pairs of poses whose shared substructure sits in
the same place in Cartesian space."""

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS

from openplaceholder.core.selection.objective import Objective, ObjectiveConfig
from openplaceholder.core.structure import Structure

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=True)
class ScaffoldRMSDObjectiveConfig(ObjectiveConfig):
    # RMSD (A) at which a pair scores 1/e; smaller demands tighter agreement
    tolerance: float = 1.0
    # SMARTS for the shared scaffold; when unset it is derived from the pool
    core_smarts: str | None = None
    # a core smaller than this is not a scaffold; below it the objective
    # switches itself off rather than scoring noise (see _resolve_core)
    min_core_atoms: int = 6
    # seconds allowed for the one maximum common substructure search
    mcs_timeout: int = 30
    # cap on symmetry-equivalent core matchings kept per structure
    max_matches: int = 64


class ScaffoldRMSDObjective(Objective):
    """In-place RMSD of two poses' shared scaffold.
    Deliberately *not* superimposed: the normalizer already puts every complex
    in one frame.
    """

    _config: ScaffoldRMSDObjectiveConfig

    def __init__(self, config: ScaffoldRMSDObjectiveConfig):
        super().__init__(config)
        self._core: Chem.Mol | None = None
        self._resolved = False
        self._coords: dict[Structure, list[np.ndarray]] = {}

    def matrix(self, structures: Sequence[Structure]) -> np.ndarray:
        self._core = self._resolve_core(structures)
        self._resolved = True
        self._coords = {}
        return super().matrix(structures)

    def score(self, a: Structure, b: Structure) -> float:
        core = self._core if self._resolved else self._pair_core(a, b)
        if core is None:
            return 0.0

        reference, alternatives = self._core_coords(a, core), self._core_coords(b, core)
        if not reference or not alternatives:
            return 0.0

        # a symmetric core matches several ways; a flipped ring is not a
        # displacement, so take the best correspondence
        rmsd = min(float(np.sqrt(np.mean(np.sum((reference[0] - other) ** 2, axis=1)))) for other in alternatives)
        return float(np.exp(-((rmsd / self._config.tolerance) ** 2)))

    def _core_coords(self, structure: Structure, core: Chem.Mol) -> list[np.ndarray]:
        """Core atom positions, one entry per symmetry-equivalent matching."""
        if structure not in self._coords:
            mol = Chem.RemoveHs(structure.to_rdkit_ligand_mol())
            matches = mol.GetSubstructMatches(core, uniquify=False, maxMatches=self._config.max_matches)
            positions = mol.GetConformer().GetPositions()
            self._coords[structure] = [positions[list(match)] for match in matches]
        return self._coords[structure]

    def _resolve_core(self, structures: Sequence[Structure]) -> Chem.Mol | None:
        if self._config.core_smarts:
            return Chem.MolFromSmarts(self._config.core_smarts)

        smiles = sorted({structure.ligand_smiles for structure in structures})
        core = self._mcs([Chem.MolFromSmiles(s) for s in smiles])
        if core is None:
            logger.warning("no common scaffold across %d ligands; scores will be 0", len(smiles))
            return None

        atoms = core.GetNumAtoms()
        if atoms < self._config.min_core_atoms:
            # diversity can result in tiny MCS - not useful
            logger.warning(
                "shared scaffold across %d ligands is only %d atoms (min %d): "
                "the ligands are too dissimilar for %s to mean anything, so it is "
                "contributing nothing. Set core_smarts to pin a scaffold, or drop "
                "this objective for this set.",
                len(smiles),
                atoms,
                self._config.min_core_atoms,
                type(self).__name__,
            )
            return None

        logger.info("scaffold core: %d atoms across %d ligands", atoms, len(smiles))
        return core

    def _pair_core(self, a: Structure, b: Structure) -> Chem.Mol | None:
        return self._mcs([Chem.MolFromSmiles(a.ligand_smiles), Chem.MolFromSmiles(b.ligand_smiles)])

    def _mcs(self, mols: list[Chem.Mol]) -> Chem.Mol | None:
        if any(mol is None for mol in mols):
            return None
        if len(mols) == 1:
            return mols[0]
        result = rdFMCS.FindMCS(
            mols,
            matchValences=False,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            matchChiralTag=False,
            timeout=self._config.mcs_timeout,
        )
        if result.canceled or not result.smartsString:
            return None
        return Chem.MolFromSmarts(result.smartsString)
