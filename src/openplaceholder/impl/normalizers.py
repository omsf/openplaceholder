"""Normalizers that put candidate structures on a common footing."""

import logging
from dataclasses import dataclass

from MDAnalysis import Universe
from MDAnalysis.analysis import align

from openplaceholder.core.selection.normalizer import Normalizer, NormalizerConfigBase
from openplaceholder.core.structure import Structure, StructureSet

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BindingSiteAlignerConfig(NormalizerConfigBase):
    # CA atoms this close to the reference ligand make up the fit
    radius: float = 6.0


class BindingSiteAligner(Normalizer):
    """Superpose every complex onto a reference by its binding site CA atoms.

    Each prediction comes out in its own arbitrary frame, so poses from
    different complexes only become comparable once they share one. Fitting on
    the pocket rather than the whole fold keeps the error where it matters:
    the global fold is predicted consistently enough that including it dilutes
    the binding site. The rigid transform is applied to the whole complex, so
    every ligand keeps its co-folded pose relative to its own protein.
    """

    _config: BindingSiteAlignerConfig

    def _setup(self) -> None:
        pass

    def _normalize(self, structures: list[StructureSet]) -> list[StructureSet]:
        reference = structures[0][0].to_mda_universe()
        site = reference.select_atoms(f"name CA and around {self._config.radius} (not protein and not water)")
        site_selection = f"name CA and resid {' '.join(str(resid) for resid in site.resids)}"
        logger.info("aligning on %d binding site CA atoms", len(site))

        return [
            StructureSet.from_structures([self._superpose(s, reference, site_selection) for s in structure_set])
            for structure_set in structures
        ]

    @staticmethod
    def _superpose(structure: Structure, reference: Universe, site_selection: str) -> Structure:
        # on a copy: to_mda_universe is cached, and alignto moves atoms in place
        universe = structure.to_mda_universe().copy()
        align.alignto(universe, reference, select=site_selection)
        return structure.with_atoms(universe.atoms)
