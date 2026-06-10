from dataclasses import dataclass

from MDAnalysis.guesser.tables import vdwradii
from MDAnalysis.lib.distances import self_capped_distance
from posebusters import PoseBusters
from rdkit import Chem
from rdkit.Chem import AllChem, inchi

from openplaceholder.core.selection.validator import Validator
from openplaceholder.core.structure import Structure


@dataclass(frozen=True, eq=True)
class PosebustersValidatorConfig:
    # the maximum number of allowed PoseBusters violations before a pose is considered flawed
    max_violations: int = 0


class PosebustersValidator(Validator):
    """Reject poses that fail more than `max_failures` of the configured PoseBusters checks.

    Passing generated molecules conformations will have reasonable geometries,
    have standard bond lengths and angles and no intramolecular steric clashes.
    """

    def __init__(self, config: PosebustersValidatorConfig):
        self._config = config
        self._buster = PoseBusters(config="mol")

    def _validate_structure(self, structure: Structure) -> bool:
        return self._count_violations(structure) <= self._config.max_violations

    def _count_violations(self, structure: Structure) -> int:
        ligand = structure.to_mda_universe().select_atoms(f"resname {structure.ligand_name}").convert_to("RDKIT")
        report = self._buster.bust(mol_pred=ligand)  # can use this later on for granular validation
        violations = 0
        for check, passed in report.iloc[0].items():
            if not passed:
                violations += 1
        return violations


@dataclass(frozen=True, eq=True)
class StereoValidatorConfig:
    # require the standard InChI of the cofolded ligand to match the requested ligand
    require_inchi_match: bool = True
    # require the canonical isomeric SMILES of the cofolded ligand to match the requested ligand
    require_smiles_match: bool = True


class StereoValidator(Validator):
    """Reject poses whose cofolded ligand stereochemistry drifts from the requested ligand.

    The requested ligand is taken from its SMILES; the cofolded ligand has its bond orders
    assigned from that template (so connectivity is shared) and its stereochemistry perceived
    independently from the 3D coordinates. The two are then compared as canonical InChI and/or
    canonical isomeric SMILES. InChI is a toolkit-independent, tautomer-normalised reference,
    while SMILES is stricter and also captures enhanced stereochemistry InChI cannot represent;
    requiring both to agree guards against the blind spots of either representation.
    """

    def __init__(self, config: StereoValidatorConfig):
        self._config = config

    def _validate_structure(self, structure: Structure) -> bool:
        reference = Chem.MolFromSmiles(structure.ligand_smiles)
        cofolded = self._cofolded_mol(structure)
        checks = []
        if self._config.require_inchi_match:
            checks.append(self._same_inchi(reference, cofolded))
        if self._config.require_smiles_match:
            checks.append(self._same_smiles(reference, cofolded))
        return all(checks)

    def _cofolded_mol(self, structure: Structure) -> Chem.Mol:
        ligand = structure.to_mda_universe().select_atoms(f"resname {structure.ligand_name}")
        # cofolded ligands carry no explicit hydrogens, so force the conversion past that check and
        # recover the heavy-atom skeleton; bond orders and hydrogens come from the template below.
        mol = ligand.convert_to("RDKIT", force=True)
        template = Chem.MolFromSmiles(structure.ligand_smiles)
        mol = AllChem.AssignBondOrdersFromTemplate(template, mol)
        # the forced conversion left atoms flagged as radicals with implicit hydrogens suppressed;
        # clear that so sanitisation fills valences (and hydrogen counts) from the assigned bond orders.
        editable = Chem.RWMol(mol)
        for atom in editable.GetAtoms():
            atom.SetNoImplicit(False)
            atom.SetNumExplicitHs(0)
            atom.SetNumRadicalElectrons(0)
        mol = editable.GetMol()
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistryFrom3D(mol)
        return mol

    @staticmethod
    def _same_inchi(reference: Chem.Mol, cofolded: Chem.Mol) -> bool:
        return inchi.MolToInchi(reference) == inchi.MolToInchi(cofolded)

    @staticmethod
    def _same_smiles(reference: Chem.Mol, cofolded: Chem.Mol) -> bool:
        return Chem.MolToSmiles(reference) == Chem.MolToSmiles(cofolded)


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


@dataclass(frozen=True, eq=True)
class SequenceValidatorConfig:
    # maximum number of residues allowed to differ from the requested sequence
    max_mismatches: int = 0


class SequenceValidator(Validator):
    """Reject poses whose modelled protein sequence drifts from the requested amino-acid sequence."""

    def __init__(self, config: SequenceValidatorConfig):
        self._config = config

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
