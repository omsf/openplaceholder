"""Structure transformations that modify co-folded complexes for FEP."""

import io
import logging
import math
import re
from dataclasses import dataclass

import MDAnalysis as mda
import numpy as np
import openmm
from MDAnalysis.analysis import align
from openff.toolkit import Molecule
from openff.toolkit.utils.exceptions import OpenFFToolkitException
from openff.units import unit as off_unit
from openmm import unit as omm_unit
from openmm.app import ForceField, Modeller, NoCutoff, PDBFile, Topology
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from pdbfixer import PDBFixer
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial import ConvexHull

from openplaceholder.core.assembly.transformation import (
    Transformation,
    TransformationConfigBase,
)
from openplaceholder.core.structure import (
    LigandPerceptionError,
    Structure,
    StructureSeries,
    atoms_to_pdb_string,
)
from openplaceholder.vendor.protonate_utils import (
    protonate_molecule,
    protonate_structure,
)

logger = logging.getLogger(__name__)

# TODO: this should be moved to a higher level
# stable PDB residue name for the protonated ligand -- RDKit's MolToPDBBlock
# stamps "UNL" otherwise, and a PDB resName is only three characters (so a
# longer ligand_name cannot be preserved here)
_LIGAND_RESNAME = "LIG"

# TODO: does this need to be here and not just where it was used?
# hydride's compiled relaxation step (geometry-optimising the placed hydrogens)
# is incompatible with the numpy 2.x stack here (an int32/long buffer mismatch);
# hydrogen *placement* -- the pH-correct states -- is unaffected, so relaxation
# is left off.
_RELAX_PROTEIN_HYDROGENS = False


def _ligand_volume(structure: Structure) -> float:
    return float(ConvexHull(structure.ligand_atoms().positions).volume)


def _rebuild(structure: Structure, protein: mda.AtomGroup, ligand: mda.AtomGroup) -> Structure:
    """Reassemble a complex from its protein + ligand atoms, keeping metadata."""
    return structure.with_atoms(mda.Merge(protein, ligand).atoms)


@dataclass(frozen=True)
class MaxVolumeSiteSubstitutionTransformationConfig(TransformationConfigBase):
    pass


class MaxVolumeSiteSubstitutionTransformation(Transformation):
    """Give every complex the same (canonical) protein coordinates.

    Picks the complex whose ligand occupies the largest convex-hull
    volume as the canonical protein, superposes every complex onto the
    protein CA, and rebuilds each structure with new canonical
    protein.
    """

    _config: MaxVolumeSiteSubstitutionTransformationConfig

    def _setup(self) -> None:
        pass

    def _transform(self, structures: StructureSeries) -> StructureSeries:

        if len(structures) == 1:
            return structures

        # TODO: would be cleaner if we supported __getitem__
        flat_structures = list(structures.iter_series())
        reference_structure = flat_structures[0]
        reference_volume = _ligand_volume(reference_structure)
        for s in flat_structures[1:]:
            if (_volume := _ligand_volume(s)) > reference_volume:
                reference_volume = _volume
                reference_structure = s
        u_reference = reference_structure.to_mda_universe()
        canonical_protein = u_reference.select_atoms("protein")
        return StructureSeries([self._substitute(s, u_reference, canonical_protein) for s in flat_structures])

    @staticmethod
    def _substitute(structure: Structure, reference: mda.Universe, canonical_protein: mda.AtomGroup) -> Structure:
        # TODO: only apply rotation and translation to the ligand
        # Protein CA align system to reference and swap protein coordinates
        mobile = structure.to_mda_universe()
        align.alignto(mobile, reference, select="protein and name CA")
        return _rebuild(structure, canonical_protein, mobile.select_atoms("not protein"))


@dataclass(frozen=True)
class HeavyAtomAdditionTransformationConfig(TransformationConfigBase):
    pass


class HeavyAtomAdditionTransformation(Transformation):
    """Add missing heavy atoms to each complex's protein with PDBFixer."""

    _config: HeavyAtomAdditionTransformationConfig

    def _setup(self) -> None:
        pass

    def _transform(self, structures: StructureSeries) -> StructureSeries:
        return StructureSeries([self._prepare_protein(s) for s in structures.iter_series()])

    def _prepare_protein(self, structure: Structure) -> Structure:

        with io.StringIO(atoms_to_pdb_string(structure.protein_atoms())) as buffer:
            fixer = PDBFixer(pdbfile=buffer)

        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()

        with io.StringIO() as buffer:
            PDBFile.writeFile(fixer.topology, fixer.positions, buffer)
            u_heavy_atoms = mda.Universe(buffer, topology_format="PDB")

        return _rebuild(structure, u_heavy_atoms.atoms, structure.ligand_atoms())


@dataclass(frozen=True)
class ComplexProtonationTransformationConfig(TransformationConfigBase):
    ph: float = 7.0  # protonation pH for both protein and ligand


class ComplexProtonationTransformation(Transformation):
    """Protonate every complex's ligand and protein at pH = ``ph``.

    Ligand protonation uses Dimorphite-DL microstates and protein
    protonation uses hydride (both vendored from
    PatWalters/protonate_utils). Protein and ligand protonation are
    independent (the ligand method never sees the protein and vice
    versa), so the ligand-protein interface protonation is not
    guaranteed self-consistent.
    """

    _config: ComplexProtonationTransformationConfig

    def _setup(self) -> None:
        pass

    def _transform(self, structures: StructureSeries) -> StructureSeries:
        return StructureSeries([self._protonate_protein(self._protonate_ligand(s)) for s in structures.iter_series()])

    def _protonate_ligand(self, structure: Structure) -> Structure:
        mol: Chem.Mol = protonate_molecule(structure.to_rdkit_ligand_mol(), self._config.ph)  # type: ignore[no-untyped-call]

        # ensure the ligand has the right name after writing
        for atom in mol.GetAtoms():  # type: ignore
            info = Chem.AtomPDBResidueInfo()
            info.SetResidueName(_LIGAND_RESNAME)
            info.SetIsHeteroAtom(True)
            atom.SetMonomerInfo(info)

        with io.StringIO(Chem.MolToPDBBlock(mol)) as buffer:
            ligand = mda.Universe(buffer, topology_format="PDB").atoms

        return _rebuild(structure, structure.protein_atoms(), ligand)

    def _protonate_protein(self, structure: Structure) -> Structure:
        import biotite.structure.io.pdb as pdb_io

        protein_pdb_string = atoms_to_pdb_string(structure.protein_atoms())
        with io.StringIO(protein_pdb_string) as buffer:
            source = pdb_io.PDBFile.read(buffer)

        protonated = protonate_structure(  # type: ignore[no-untyped-call]
            source.get_structure(model=1), ph=self._config.ph, relax=_RELAX_PROTEIN_HYDROGENS
        )

        out_file = pdb_io.PDBFile()
        out_file.set_structure(protonated)
        with io.StringIO() as buffer:
            out_file.write(buffer)
            reconstructed = mda.Universe(buffer, topology_format="PDB")

        return _rebuild(structure, reconstructed.atoms, structure.ligand_atoms())


# an OpenMM template error names the offending residue by its topology index,
# e.g. "No template found for residue 35 (HIS)"
_TEMPLATE_ERROR_RESIDUE = re.compile(r"residue (\d+)")

# the minimizer needs a Context and a Context needs an integrator; it is never
# stepped, so the timestep only has to be a valid one
_SMOKETEST_TIMESTEP = 0.001 * omm_unit.picoseconds


def _energy_and_max_force(context: openmm.Context) -> tuple[float, float]:
    """Read the potential energy (kJ/mol) and the largest force on any atom (kJ/mol/nm)."""
    state = context.getState(getEnergy=True, getForces=True)
    forces = state.getForces(asNumpy=True).value_in_unit(omm_unit.kilojoule_per_mole / omm_unit.nanometer)
    return (
        float(state.getPotentialEnergy().value_in_unit(omm_unit.kilojoule_per_mole)),
        float(np.linalg.norm(forces, axis=1).max()),
    )


def _describe_residue(topology: Topology, error: Exception) -> str:
    """Translate the residue index in an OpenMM template error into a locatable residue.

    OpenMM reports ``residue 35``, which is a topology index; the useful
    diagnostic is the chain/name/number and the atom set that failed to
    match a template.
    """
    match = _TEMPLATE_ERROR_RESIDUE.search(str(error))
    residues = list(topology.residues())
    if match is None or not 0 <= (index := int(match.group(1))) < len(residues):
        return "an unidentifiable residue"
    residue = residues[index]
    atoms = sorted(atom.name for atom in residue.atoms())
    return f"chain {residue.chain.id} {residue.name} {residue.id} (atoms: {atoms})"


class ComplexSmoketestError(Exception):
    """Raised when a complex fails the smoketest."""


@dataclass(frozen=True)
class ComplexSmoketestTransformationConfig(TransformationConfigBase):
    # OpenMM force field XML(s) for the protein; the openfe RFE protocol default
    protein_forcefield: tuple[str, ...] = ("amber/ff14SB.xml",)
    # SMIRNOFF force field for the ligand; the openfe RFE protocol default
    ligand_forcefield: str = "openff-2.2.1"
    # L-BFGS iterations of the vacuum minimization
    minimization_steps: int = 100
    # largest force (kJ/mol/nm) tolerated on the pose as it arrives. an unrelaxed
    # but sane complex sits around 1e5; atoms on top of each other reach 1e7 and up
    max_initial_force: float = 1.0e6
    # largest force (kJ/mol/nm) tolerated once minimization has finished
    max_minimized_force: float = 1.0e4
    # continue with whatever passed instead of erroring on the first failing complex
    drop_failures: bool = False


class ComplexSmoketestTransformation(Transformation):
    """Refuse complexes a molecular mechanics force field cannot handle.

    Every complex is parameterised in vacuum (protein force field plus
    a SMIRNOFF template for the ligand) and put through a very short
    minimization on the OpenMM CPU platform. This is a test, not a
    preparation step: structures come out untouched and the minimized
    coordinates are thrown away. A complex fails when its ligand cannot
    be parameterised, when any residue has no matching force field
    template, when the pose arrives with forces above
    ``max_initial_force``, or when minimization does not leave it at a
    finite energy with all forces below ``max_minimized_force``.

    A single failure raises. These are systematic problems -- one
    mis-protonated residue is the same residue in every complex -- so
    the default is to stop and say so rather than quietly hand a
    smaller network to the mapper. Set ``drop_failures`` to keep the
    complexes that passed and carry on.

    This is the cheap stand-in for failure modes that otherwise only
    surface once an FEP simulation has NaN'd hours in: residue
    name/atom-set mismatches, mis-protonated or incomplete residues,
    bogus bonds across chain breaks, and steric overlap too severe to
    minimize away.

    It has to run last. Heavy atoms must already be present and the
    complex protonated -- a bare co-folded complex matches no protein
    template at all, so every structure would fail for the wrong
    reason.
    """

    _config: ComplexSmoketestTransformationConfig

    def _setup(self) -> None:
        # a smoketest is not worth a GPU context, and asking for one on a
        # headless or busy machine is a failure mode of its own
        self._platform = openmm.Platform.getPlatformByName("CPU")

    def _transform(self, structures: StructureSeries) -> StructureSeries:
        # every complex is smoketested even when the first one fails: the run
        # that got here was expensive, and reporting all of the broken ones at
        # once beats rediscovering them one pipeline invocation at a time
        survivors = []
        failures = []
        for structure in structures.iter_series():
            if (failure := self._smoketest(structure)) is None:
                survivors.append(structure)
            else:
                logger.warning("structure '%s' failed the smoketest: %s", structure.ligand_name, failure)
                failures.append(f"{structure.ligand_name}: {failure}")

        if failures and not self._config.drop_failures:
            raise ComplexSmoketestError(
                f"{len(failures)} of {len(structures)} structures failed the smoketest:"
                + "".join(f"\n  {f}" for f in failures)
                + "\nset drop_failures to keep the structures that passed and continue"
            )
        if not survivors:
            raise ComplexSmoketestError("no structures survived the smoketest")
        return StructureSeries(survivors)

    def _smoketest(self, structure: Structure) -> str | None:
        """Smoketest one complex, returning ``None`` when it passes and why it failed otherwise."""

        try:
            ligand = self._to_openff_ligand(structure)
        except (LigandPerceptionError, OpenFFToolkitException, ValueError) as exc:
            return f"ligand could not be parameterised: {exc}"

        topology, positions = self._assemble(structure, ligand)
        force_field = ForceField(*self._config.protein_forcefield)
        force_field.registerTemplateGenerator(
            SMIRNOFFTemplateGenerator(molecules=ligand, forcefield=self._config.ligand_forcefield).generator
        )

        try:
            system = force_field.createSystem(topology, nonbondedMethod=NoCutoff, constraints=None)
        except ValueError as exc:
            return f"no force field template for {_describe_residue(topology, exc)}: {exc}"

        # a smoketest minimization in vacuum leaves every atom free to move, so
        # it relaxes clashes a solvated, restrained production run cannot; the
        # severe overlap that NaNs a simulation is only visible before it runs
        context = openmm.Context(system, openmm.VerletIntegrator(_SMOKETEST_TIMESTEP), self._platform)
        context.setPositions(positions)
        _, initial_force = _energy_and_max_force(context)
        if not math.isfinite(initial_force) or initial_force > self._config.max_initial_force:
            return (
                f"the pose starts at a {initial_force:.3g} kJ/mol/nm force, above the "
                f"{self._config.max_initial_force:.3g} kJ/mol/nm tolerance -- atoms are on top of each other"
            )

        try:
            openmm.LocalEnergyMinimizer.minimize(context, maxIterations=self._config.minimization_steps)
        except openmm.OpenMMException as exc:
            return f"minimization failed: {exc}"

        energy, force = _energy_and_max_force(context)
        if not math.isfinite(energy):
            return f"minimized energy is non-finite ({energy} kJ/mol)"
        if force > self._config.max_minimized_force:
            return (
                f"minimization left a {force:.3g} kJ/mol/nm force, "
                f"above the {self._config.max_minimized_force:.3g} kJ/mol/nm tolerance"
            )

        logger.info(
            "structure '%s' passed the smoketest (%d particles, %.3g kJ/mol, max force %.3g kJ/mol/nm)",
            structure.ligand_name,
            system.getNumParticles(),
            energy,
            force,
        )
        return None

    @staticmethod
    def _to_openff_ligand(structure: Structure) -> Molecule:
        """Perceive the ligand and hand it to OpenFF with cheap placeholder charges."""
        mol = structure.to_rdkit_ligand_mol()
        if not any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):  # type: ignore
            raise ValueError("ligand carries no explicit hydrogens -- protonate the complex first")

        # Gasteiger charges stand in for the AM1BCC charges the production
        # protocol assigns: they are free, while AM1BCC would run sqm and
        # dominate the runtime without testing anything this is looking for
        AllChem.ComputeGasteigerCharges(mol)  # type: ignore[attr-defined]
        charges = np.array([atom.GetDoubleProp("_GasteigerCharge") for atom in mol.GetAtoms()])  # type: ignore

        ligand: Molecule = Molecule.from_rdkit(mol, allow_undefined_stereo=True)
        ligand.partial_charges = charges * off_unit.elementary_charge
        return ligand

    @staticmethod
    def _assemble(structure: Structure, ligand: Molecule) -> tuple[Topology, omm_unit.Quantity]:
        """Merge the protein and the perceived ligand into one OpenMM topology."""
        with io.StringIO(atoms_to_pdb_string(structure.protein_atoms())) as buffer:
            protein = PDBFile(buffer)

        # the ligand topology comes from the perceived molecule rather than the
        # PDB: a HETATM block carries no bond orders and OpenMM would only bond
        # the ligand through CONECT records
        modeller = Modeller(protein.topology, protein.positions)
        modeller.add(ligand.to_topology().to_openmm(), ligand.conformers[0].to_openmm())
        return modeller.topology, modeller.positions
