"""Shared-volume objective: rewards pairs of ligand poses that occupy the
same region of 3D space."""

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError

from openplaceholder.core.selection.objective import Objective, ObjectiveConfig
from openplaceholder.core.structure import Structure


@dataclass(frozen=True, eq=True)
class VolumeOverlapObjectiveConfig(ObjectiveConfig):
    pass


class VolumeOverlapObjective(Objective):

    _config: VolumeOverlapObjectiveConfig

    def __init__(self, config: VolumeOverlapObjectiveConfig):
        super().__init__(config)
        # matrix() calls score() once per *pair*; caching each structure's
        # hull keeps that one-time cost to O(n) rather than O(n^2).
        self._hull_cache: dict[Structure, ConvexHull] = {}

    def score(self, a: Structure, b: Structure) -> float:
        """Jaccard overlap (intersection / union) of the two ligands' convex hull volumes."""
        hull_a = self._hull(a)
        hull_b = self._hull(b)

        intersection = self._intersection_volume(hull_a, hull_b)
        union = hull_a.volume + hull_b.volume - intersection
        return intersection / union if union > 0 else 0.0

    def _hull(self, structure: Structure) -> ConvexHull:
        if structure not in self._hull_cache:
            self._hull_cache[structure] = ConvexHull(self._ligand_coords(structure))
        return self._hull_cache[structure]

    @staticmethod
    def _ligand_coords(structure: Structure) -> np.ndarray:
        ligand = structure.to_mda_universe().select_atoms(f"resname {structure.ligand_name}")

        # Built directly from atoms + positions rather than via
        # AtomGroup.convert_to("RDKIT"): that path guesses bonds, which is
        # both unnecessary (only coordinates are needed here) and unreliable
        # for predicted poses (e.g. missing hydrogens, non-element atom names).
        mol = Chem.RWMol()
        conformer = Chem.Conformer(len(ligand))
        for i, atom in enumerate(ligand):
            mol.AddAtom(Chem.Atom(atom.element))
            conformer.SetAtomPosition(i, Point3D(*atom.position.astype(float)))
        mol.AddConformer(conformer)

        return np.array(mol.GetConformer().GetPositions())

    @classmethod
    def _intersection_volume(cls, hull_a: ConvexHull, hull_b: ConvexHull) -> float:
        halfspaces = np.vstack([hull_a.equations, hull_b.equations])
        interior_point = cls._chebyshev_center(halfspaces)
        if interior_point is None:
            return 0.0  # hulls don't overlap

        try:
            vertices = HalfspaceIntersection(halfspaces, interior_point).intersections
        except QhullError:
            return 0.0
        return ConvexHull(vertices).volume if len(vertices) >= 4 else 0.0

    @staticmethod
    def _chebyshev_center(halfspaces: np.ndarray) -> np.ndarray | None:
        """Point strictly inside the intersection of ``halfspaces`` (each row
        ``[a, b]`` encodes ``a @ x + b <= 0``), or None if infeasible.

        Maximizes the radius of a ball centered at ``x`` that still satisfies
        every constraint: ``normals @ x + offsets + r * ||normals|| <= 0``.
        """
        normals, offsets = halfspaces[:, :-1], halfspaces[:, -1]
        norms = np.linalg.norm(normals, axis=1)

        c = np.zeros(normals.shape[1] + 1)
        c[-1] = -1  # maximize r == minimize -r
        a_ub = np.hstack([normals, norms[:, None]])
        bounds = [(None, None)] * normals.shape[1] + [(1e-9, None)]

        result = linprog(c, A_ub=a_ub, b_ub=-offsets, bounds=bounds)
        return result.x[:-1] if result.success else None
