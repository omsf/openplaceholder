"""Normalizer implementations."""

import logging
from dataclasses import dataclass

from MDAnalysis.analysis import align

from openplaceholder.core.selection.normalizer import Normalizer, NormalizerConfigBase
from openplaceholder.core.structure import Structure, StructureSet

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BindingSiteAlignerConfig(NormalizerConfigBase):
    # CA atoms this close to the reference ligand make up the fit
    radius: float = 6.0


class BindingSiteAligner(Normalizer):
    """Superimpose each complex onto a common reference by its binding
    site CA atoms. The first Structure in the first StructureReplicate
    is chosen as the reference to which all other Structures will be
    aligned.
    """

    _config: BindingSiteAlignerConfig

    def _setup(self) -> None:
        pass

    def _normalize(self, structures: StructureSet) -> StructureSet:
        # define initial alignment parameters that will apply to all structures in the set
        reference = next(structures.iter_replicates()).replicates[0].to_mda_universe()
        site_resids = reference.select_atoms(
            f"name CA and around {self._config.radius} (not protein and not water)"
        ).resids
        site_selection = f"name CA and resid {' '.join(map(str, site_resids))}"

        def superimpose(structure: Structure) -> Structure:
            universe = structure.to_mda_universe().copy()
            align.alignto(universe, reference, select=site_selection)
            return structure.with_atoms(universe.atoms)

        logger.info("aligning on %d binding site CA atoms", len(site_resids))
        normalized_structures: list[list[Structure]] = []
        for replicates in structures.iter_replicates():
            normalized_structures.append(list(map(superimpose, replicates.iter_replicates())))

        return StructureSet.from_structures(normalized_structures)
