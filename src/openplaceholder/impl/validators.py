from dataclasses import dataclass

from MDAnalysis.guesser.tables import vdwradii
from MDAnalysis.lib.distances import self_capped_distance
from posebusters import PoseBusters

from openplaceholder.core.selection.validator import Validator
from openplaceholder.core.structure.structure import Structure


@dataclass(frozen=True, eq=True)
class PosebustersValidatorConfig:
    # PoseBusters preset to run: "mol" (ligand validity), "dock", "gen", ...
    preset: str = "mol"
    # maximum number of failed PoseBusters checks tolerated before a pose is rejected
    max_failures: int = 0


class PosebustersValidator(Validator):
    """Reject poses that fail more than `max_failures` of the configured PoseBusters checks.
    
    Passing generated molecules conformations will have reasonable geometries,
    have standard bond lengths and angles and no intramolecular steric clashes.
    """

    def __init__(self, config: PosebustersValidatorConfig):
        self._config = config
        self._buster = PoseBusters(config=config.preset)

    def _validate_structure(self, structure: Structure) -> bool:
        failures = sum(1 for passed in self.results(structure).values() if not passed)
        return failures <= self._config.max_failures

    def results(self, structure: Structure) -> dict[str, bool]:
        ligand = structure.to_mda().select_atoms(f"resname {structure.ligand_name}").convert_to("RDKIT")
        report = self._buster.bust(mol_pred=ligand) # can use this later on for granular validation
        return {check: bool(passed) for check, passed in report.iloc[0].items()}


@dataclass(frozen=True, eq=True)
class ClashValidatorConfig:
    # a non-bonded atom pair clashes when closer than this fraction of their summed van der Waals radii
    clash_tolerance: float = 0.63
    # only atoms within this distance (angstrom) of the ligand are considered the binding site
    site_radius: float = 10.0
    # the maximum number of allowed clashes before a pose is considered flawed
    max_clashes: int = 0


class ClashValidator(Validator):
    """Reject poses with too many steric clashes between non-bonded atoms in the binding site.
    
    We do this ourselves rather than with `PosebustersValidator` to allow more fine controls.
    """

    def __init__(self, config: ClashValidatorConfig):
        self._config = config

    def _validate_structure(self, structure: Structure) -> bool:
        return self._count_clashes(structure) <= self._config.max_clashes

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
