import base64
import gzip
import random
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from gufe import (
    ChemicalSystem,
    ProteinComponent,
    SmallMoleculeComponent,
    SolventComponent,
    Transformation,
)
from gufe.protocols import Protocol, ProtocolDAGResult, ProtocolUnit
from gufe.protocols.protocol import ProtocolResult
from gufe.protocols.protocolunit import ProtocolUnitFailure, ProtocolUnitResult
from gufe.settings.models import Settings
from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
from openfe.protocols.openmm_rfe.equil_rfe_settings import (
    RelativeHybridTopologyProtocolSettings,
)
from openff.units import unit
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule

from openplaceholder.core.simulation.simulator import SimulationResults
from openplaceholder.core.structure import Structure, StructureFormat

DEFAULT_SEQUENCE = "MGSPASDPTVFHKRYLKKIRDLGEGHFGKVSLYCYDPTNDGTGEMVAVKALKADAGPQHRSGWKQEIDILRTLYHEHIIKYKGCCEDAGAASLQLVMEYVPLGSLRDYLPRHSIGLAQLLLFAQQICEGMAYLHAQHYIHRNLAARNVLLDNDRLVKIGDFGLAKAVPEGHEYYRVREDGDSPVFWYAPECLKEYKFYYASDVWSFGVTLYELLTHCDSSQSPPTKFLELIGIAQGQMTVLRLTELLERGERLPRPDKCPAEVYHLMKNCWETEASFRPTFENLIPILKTVHEKYRHHHHHH"
DEFAULT_SMILES = "C1=CC=CC=C1"
DEFAULT_LIGAND_NAME = "benzene"
DEFAULT_FORMAT = StructureFormat.PDB


def make_structures(
    n_structures: int = 25,
    sequence: str = DEFAULT_SEQUENCE,
    ligand_smiles: str = DEFAULT_SMILES,
    ligand_name: str = DEFAULT_LIGAND_NAME,
    structure_format: str = DEFAULT_FORMAT,
) -> list[Structure]:
    template = Structure(
        sequence=sequence,
        ligand_smiles=ligand_smiles,
        ligand_name=ligand_name,
        structure_format=structure_format,
        structure_data="",
    )
    return [
        template.copy_with_replacements(structure_data=base64.b64encode(random.randbytes(8)).decode())
        for _ in range(n_structures)
    ]


def read_gzip_file(file_path: str | Path) -> bytes:
    file_path = Path(file_path)

    with gzip.open(file_path, "r") as gz:
        content = gz.read()

    return content


# --- analyzer fixtures ----------------------------------------------------
#
# An analyzer only ever asks a transformation's protocol to gather a result and
# report an estimate, so the stubs below stand in for a simulated leg: the
# estimate is supplied rather than computed, which keeps the free energies of a
# test network exactly known.

TINY_PROTEIN_PDB = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.988  -0.773   1.199  1.00  0.00           C
TER
END
"""

KCAL = unit.kilocalorie_per_mole


class StubProtocolResult(ProtocolResult):  # type: ignore
    """A ``ProtocolResult`` that reports the estimate it was handed."""

    def get_estimate(self) -> Any:
        return self.data["estimate"]

    def get_uncertainty(self) -> Any:
        return self.data["uncertainty"]

    def get_individual_estimates(self) -> Any:
        return self.data["individual"]


class StubProtocolUnit(ProtocolUnit):  # type: ignore
    """A unit that is never executed, as its result is supplied directly."""

    def _execute(self, ctx: Any, **inputs: Any) -> dict[str, Any]:
        raise NotImplementedError


class StubProtocol(Protocol):  # type: ignore
    """A protocol that reports whatever estimate its unit results carry."""

    result_cls = StubProtocolResult
    _settings_cls = RelativeHybridTopologyProtocolSettings

    @classmethod
    def _default_settings(cls) -> Settings:
        return RelativeHybridTopologyProtocol.default_settings()

    def _create(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: Any = None,
        extends: ProtocolDAGResult | None = None,
    ) -> list[ProtocolUnit]:
        _, _, _, _ = stateA, stateB, mapping, extends
        return [StubProtocolUnit(name="stub")]

    def _gather(self, protocol_dag_results: Iterable[ProtocolDAGResult]) -> dict[str, Any]:
        outputs = [result.outputs for dag in protocol_dag_results for result in dag.protocol_unit_results]
        return {key: outputs[0][key] for key in ("estimate", "uncertainty", "individual")}


@lru_cache(maxsize=1)
def _tiny_protein() -> ProteinComponent:
    """Five atoms of alanine, enough to mark a system as the complex leg."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tiny.pdb"
        path.write_text(TINY_PROTEIN_PDB)
        return ProteinComponent.from_pdb_file(str(path), name="tiny")


@lru_cache(maxsize=None)
def _named_ligand(name: str) -> SmallMoleculeComponent:
    """Benzene under ``name``; only the name matters to an analyzer."""
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    EmbedMolecule(mol, randomSeed=42)
    return SmallMoleculeComponent(mol, name=name)


def _state(ligand: str, complexed: bool) -> ChemicalSystem:
    components: dict[str, Any] = {"ligand": _named_ligand(ligand), "solvent": SolventComponent()}
    if complexed:
        components["protein"] = _tiny_protein()
    return ChemicalSystem(components)


def make_leg(
    ligand_a: str,
    ligand_b: str,
    phase: str,
    free_energy: float,
    repeat_std: float = 0.0,
    mbar_error: float = 0.2,
    ok: bool = True,
) -> tuple[Transformation, ProtocolDAGResult]:
    """One simulated leg, reporting ``free_energy`` in kcal/mol.

    ``repeat_std`` left at zero mirrors a single repeat, where the protocol has
    no spread to report and the analyzer falls back to ``mbar_error``.
    """
    complexed = phase == "complex"
    transformation = Transformation(
        stateA=_state(ligand_a, complexed),
        stateB=_state(ligand_b, complexed),
        protocol=StubProtocol(StubProtocol.default_settings()),
        mapping=None,
        name=f"{ligand_a}_{ligand_b}_{phase}",
    )

    stub = StubProtocolUnit(name="stub")
    outputs = {
        "estimate": free_energy * KCAL,
        "uncertainty": repeat_std * KCAL,
        "individual": [(free_energy * KCAL, mbar_error * KCAL)],
    }
    result: ProtocolUnitResult | ProtocolUnitFailure
    if ok:
        result = ProtocolUnitResult(source_key=stub.key, inputs={}, outputs=outputs)
    else:
        result = ProtocolUnitFailure(
            source_key=stub.key,
            inputs={},
            outputs={},
            exception=("RuntimeError", ("leg failed",)),
            traceback="tb",
        )

    dag_result = ProtocolDAGResult(
        protocol_units=[stub],
        protocol_unit_results=[result],
        transformation_key=transformation.key,
    )
    return transformation, dag_result


def make_simulation_results(
    edges: Iterable[tuple[str, str, float, float]],
    failed: Iterable[str] = (),
    **leg_options: Any,
) -> SimulationResults:
    """A ready to analyze ``SimulationResults`` with known free energies.

    Parameters
    ----------
    edges
        ``(ligand_a, ligand_b, complex_dg, solvent_dg)`` per ligand pair, in
        kcal/mol, so the relative free energy of the pair is exactly their
        difference.
    failed
        Leg names to mark as failed, each ``f"{ligand_a}_{ligand_b}_{phase}"``.
    leg_options
        Passed through to ``make_leg``, e.g. ``repeat_std``.
    """
    failed = set(failed)
    legs = []
    for ligand_a, ligand_b, complex_dg, solvent_dg in edges:
        for phase, free_energy in (("complex", complex_dg), ("solvent", solvent_dg)):
            name = f"{ligand_a}_{ligand_b}_{phase}"
            legs.append(make_leg(ligand_a, ligand_b, phase, free_energy, ok=name not in failed, **leg_options))

    return SimulationResults(legs)
