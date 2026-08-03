import logging
from dataclasses import dataclass
from functools import cache

from MDAnalysis.guesser.tables import vdwradii
from MDAnalysis.lib.distances import self_capped_distance
from posebusters import PoseBusters
from rdkit import Chem

from openplaceholder.core.selection.validator import Validator, ValidatorConfigBase
from openplaceholder.core.structure import LigandPerceptionError, Structure

logger = logging.getLogger(__name__)


@cache
def _perceive_ligand(structure: Structure) -> Chem.Mol | None:
    """Reconstruct the ligand mol, or None (with a warning) when a distorted
    pose can't be perceived -- so validators drop it instead of crashing down the road."""

    try:
        return structure.to_rdkit_ligand_mol()
    except LigandPerceptionError as exc:
        # TODO: should this return None or just raise an
        # exception. Let downstream handle it.
        logger.warning("dropping structure unperceivable by RDKit: %s", exc)
        return None


@dataclass(frozen=True)
class PosebustersValidatorConfig(ValidatorConfigBase):
    # the maximum number of allowed PoseBusters violations before a pose is considered flawed
    max_violations: int = 0


class PosebustersValidator(Validator):
    """Reject poses that fail more than `max_failures` of the configured PoseBusters checks.

    Passing generated molecules conformations will have reasonable geometries,
    have standard bond lengths and angles and no intramolecular steric clashes.
    """

    _config: PosebustersValidatorConfig

    def _setup(self) -> None:
        self._buster = PoseBusters(config="mol")

    def _validate_structure(self, structure: Structure) -> bool:
        ligand = _perceive_ligand(structure)
        if ligand is None:
            return False
        return self._count_violations(ligand) <= self._config.max_violations

    def _count_violations(self, ligand: Chem.Mol) -> int:
        report = self._buster.bust(mol_pred=ligand)  # can use this later on for granular validation
        violations = 0
        for check, passed in report.iloc[0].items():
            if not passed:
                violations += 1
        return violations


@dataclass(frozen=True)
class StereoValidatorConfig(ValidatorConfigBase):
    # require the standard InChI of the predicted ligand to match the requested ligand
    require_inchi_match: bool = True
    # require the canonical isomeric SMILES of the predicted ligand to match the requested ligand
    require_smiles_match: bool = True


class StereoValidator(Validator):
    """Reject poses whose predicted ligand stereochemistry drifts from
    the requested ligand.

    The requested ligand is taken from its SMILES; the predicted
    ligand has its bond orders assigned from that template (so
    connectivity is shared) and its stereochemistry perceived
    independently from the 3D coordinates. The two are then compared
    as canonical InChI and/or canonical isomeric SMILES. InChI is a
    toolkit-independent, tautomer-normalised reference, while SMILES
    is stricter and also captures enhanced stereochemistry InChI
    cannot represent; requiring both to agree guards against the blind
    spots of either representation.

    """

    _config: StereoValidatorConfig

    def _setup(self) -> None:
        pass

    def _validate_structure(self, structure: Structure) -> bool:
        predicted = _perceive_ligand(structure)
        if predicted is None:
            return False
        reference = Chem.MolFromSmiles(structure.ligand_smiles)

        if self._config.require_inchi_match and not self._same_inchi(reference, predicted):
            return False
        if self._config.require_smiles_match and not self._same_smiles(reference, predicted):
            return False
        return True

    @staticmethod
    def _same_inchi(mol_a: Chem.Mol, mol_b: Chem.Mol) -> bool:
        inchi_a: str = Chem.inchi.MolToInchi(mol_a)  # type: ignore[no-untyped-call]
        inchi_b: str = Chem.inchi.MolToInchi(mol_b)  # type: ignore[no-untyped-call]
        return inchi_a == inchi_b

    @staticmethod
    def _same_smiles(mol_a: Chem.Mol, mol_b: Chem.Mol) -> bool:
        return Chem.MolToSmiles(mol_a) == Chem.MolToSmiles(mol_b)


@dataclass(frozen=True)
class ClashValidatorConfig(ValidatorConfigBase):
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

    _config: ClashValidatorConfig

    def _setup(self) -> None:
        pass

    def _validate_structure(self, structure: Structure) -> bool:
        return self._count_clashes(structure) <= self._config.max_clashes

    def _count_clashes(self, structure: Structure) -> int:
        ligand = f"resname {structure.ligand_name} or resname LIG"
        site = structure.to_mda_universe().select_atoms(f"({ligand}) or (around {self._config.site_radius} {ligand})")
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


@dataclass(frozen=True)
class SequenceValidatorConfig(ValidatorConfigBase):
    # maximum number of residues allowed to differ from the requested sequence
    max_mismatches: int = 0


class SequenceValidator(Validator):
    """Reject poses whose modelled protein sequence drifts from the requested amino-acid sequence."""

    _config: SequenceValidatorConfig

    def _validate_structure(self, structure: Structure) -> bool:
        original = structure.sequence
        modelled = structure.to_mda_universe().select_atoms("protein").residues.sequence(format="string")
        if len(original) != len(modelled):
            return False
        return self._count_mismatches(original, modelled) <= self._config.max_mismatches

    def _count_mismatches(self, original: str, modelled: str) -> int:
        substitutions = 0
        for expected, actual in zip(original, modelled):
            if expected != actual:
                substitutions += 1
        return substitutions
