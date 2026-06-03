from dataclasses import dataclass

from MDAnalysis.guesser.tables import vdwradii
from MDAnalysis.lib.distances import self_capped_distance

from openplaceholder.core.selection.validator import Validator
from openplaceholder.core.structure.structure import Structure


@dataclass(frozen=True, eq=True)
class PosebustersValidatorConfig:
    pass


class PosebustersValidator(Validator):

    def __init__(self, config: PosebustersValidatorConfig):
        self._config = config

    def validate(self, structures: list[Structure]) -> list[Structure]:
        raise NotImplementedError


@dataclass(frozen=True, eq=True)
class ClashValidatorConfig:
    # a non-bonded atom pair clashes when closer than this fraction of their summed van der Waals radii
    clash_tolerance: float = 0.63
    # only atoms within this distance (angstrom) of the ligand are considered the binding site
    site_radius: float = 10.0
    max_clashes: int = 0


class ClashValidator(Validator):
    """Reject poses with too many steric clashes between non-bonded atoms in the binding site."""

    def __init__(self, config: ClashValidatorConfig):
        self._config = config

    def validate(self, structures: list[Structure]) -> list[Structure]:
        return [s for s in structures if self._count_clashes(s) <= self._config.max_clashes]

    def _count_clashes(self, structure: Structure) -> int:
        ligand = f"resname {structure.ligand_name}"
        site = structure.to_mda().select_atoms(f"({ligand}) or (around {self._config.site_radius} {ligand})")
        radii = [vdwradii.get(element.upper(), 1.5) for element in site.elements]
        bonded = {tuple(bond) for bond in site.bonds.to_indices()}
        indices = site.ix
        tolerance = self._config.clash_tolerance
        pairs, distances = self_capped_distance(site.positions, max_cutoff=tolerance * 2 * max(radii))
        clashes = 0
        for (i, j), distance in zip(pairs, distances):
            if (indices[i], indices[j]) not in bonded and distance < tolerance * (radii[i] + radii[j]):
                clashes += 1
        return clashes
