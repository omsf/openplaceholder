from itertools import cycle, pairwise
from pathlib import Path
from typing import Any, Callable, Iterable
from unittest.mock import patch

import pytest
from gufe import (
    AlchemicalNetwork,
    ChemicalSystem,
    SmallMoleculeComponent,
    SolventComponent,
    Transformation,
)
from gufe.mapping import LigandAtomMapping
from gufe.mapping.componentmapping import ComponentMapping
from gufe.protocols import Protocol, ProtocolDAG, ProtocolDAGResult
from gufe.protocols.protocolunit import (
    ProtocolUnit,
    ProtocolUnitFailure,
    ProtocolUnitResult,
)
from gufe.settings.models import Settings
from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
from openfe.protocols.openmm_rfe.equil_rfe_settings import (
    RelativeHybridTopologyProtocolSettings,
)
from openff.units import unit
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule

from openplaceholder.core.simulation.simulator import (
    DisconnectedNetworkError,
    EmptyNetworkError,
    SimulationResults,
    UnsupportedProtocolError,
)
from openplaceholder.impl.simulators import OpenFESimulator, OpenFESimulatorConfig

PROTOCOL = RelativeHybridTopologyProtocol(settings=RelativeHybridTopologyProtocol.default_settings())


def _ligand(name: str = "benzene", smiles: str = "c1ccccc1") -> SmallMoleculeComponent:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    # constant random seed for reproducibility
    EmbedMolecule(mol, randomSeed=42)
    return SmallMoleculeComponent(mol, name=name)


def _transformation(name: str, a: str = "lig_a", b: str = "lig_b", protocol: Protocol = PROTOCOL) -> Transformation:
    """A solvent-leg RBFE transformation between two named ligands."""
    ligand_a, ligand_b = _ligand(a), _ligand(b)
    return Transformation(
        stateA=ChemicalSystem({"ligand": ligand_a, "solvent": SolventComponent()}),
        stateB=ChemicalSystem({"ligand": ligand_b, "solvent": SolventComponent()}),
        protocol=protocol,
        mapping=LigandAtomMapping(ligand_a, ligand_b, {i: i for i in range(ligand_a.to_rdkit().GetNumAtoms())}),
        name=name,
    )


def _transformation_chain(names: Iterable[str]) -> list[Transformation]:
    transformations = []
    for a, b in pairwise(names):
        transformations.append(_transformation(f"t_{a}_{b}", a, b))
    return transformations


def _dag_result(dag: ProtocolDAG, ok: bool = True, failures: bool = True) -> ProtocolDAGResult:
    source = dag.protocol_units[0]

    results: list[ProtocolUnitResult]

    match (ok, failures):
        case True, _:
            results = [ProtocolUnitResult(source_key=source.key, inputs={}, outputs={})]
        case False, True:
            results = [
                ProtocolUnitFailure(
                    source_key=source.key,
                    inputs={},
                    outputs={},
                    exception=("RuntimeError", ("boom",)),
                    traceback="tb",
                )
            ]
        case False, False:
            results = []

    return ProtocolDAGResult(
        protocol_units=[source],
        protocol_unit_results=results,
        transformation_key=dag.transformation_key,
    )


def _executes(ok: bool = True, failures: bool = True) -> Callable[[ProtocolDAG], ProtocolDAGResult]:
    """Returns a common mock function that produces ProtocolDAGResults."""

    def func(dag: ProtocolDAG, **kwargs: dict[str, Any]) -> ProtocolDAGResult:
        _ = kwargs
        return _dag_result(dag, ok=ok, failures=failures)

    return func


def _results(transformations: Iterable[Transformation], ok: bool = True) -> SimulationResults:
    results = []
    for t in transformations:
        results.append((t, _dag_result(t.create(), ok=ok)))
    return SimulationResults(results)


def _simulator(tmp_path: Path, production_length_ns: float = 5.0) -> OpenFESimulator:
    return OpenFESimulator(
        OpenFESimulatorConfig(simulation_directory=tmp_path, production_length_ns=production_length_ns)
    )


class ForeignProtocol(Protocol):  # type: ignore

    _settings_cls = RelativeHybridTopologyProtocolSettings

    @classmethod
    def _default_settings(cls) -> Settings:
        return RelativeHybridTopologyProtocol.default_settings()

    def _create(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: ComponentMapping | list[ComponentMapping] | None,
        extends: ProtocolDAGResult | None = None,
    ) -> list[ProtocolUnit]:
        _, _, _, _ = stateA, stateB, mapping, extends
        raise NotImplementedError

    def _gather(self, protocol_dag_results: Iterable[ProtocolDAGResult]) -> dict[str, Any]:
        _ = protocol_dag_results
        raise NotImplementedError


class TestSimulationResults:

    def test_transformation_result_pairs(self) -> None:
        a, b = _transformation("a", a="lig_a", b="lig_b"), _transformation("b", a="lig_b", b="lig_c")
        sim_results = _results((b, a))
        assert [transformation.name for transformation, _ in sim_results] == ["b", "a"]

    def test_ok(self) -> None:
        a = _transformation("a")

        ok = _results((a,), ok=True)
        not_ok = _results((a,), ok=False)
        empty = SimulationResults([])

        assert ok.ok()
        assert not_ok.ok() is False
        assert empty.ok() is False

    def test_round_trips_through_json(self) -> None:
        results = _results((_transformation("a"),))
        restored = SimulationResults.from_json(content=results.to_json())
        assert restored is results


class TestOpenFESimulator:

    def test_init(self, tmp_path: Path) -> None:
        _simulator(tmp_path)

    def test_default_settings(self, tmp_path: Path) -> None:
        settings = _simulator(tmp_path, production_length_ns=0.5)._protocol.settings
        defaults = RelativeHybridTopologyProtocol.default_settings()
        assert isinstance(settings, RelativeHybridTopologyProtocolSettings)
        assert isinstance(defaults, RelativeHybridTopologyProtocolSettings)
        assert settings.simulation_settings.production_length == 0.5 * unit.nanoseconds
        assert settings.forcefield_settings != defaults.forcefield_settings
        assert settings.protocol_repeats != defaults.protocol_repeats

    def test_raise_empty_network(self, tmp_path: Path) -> None:
        with pytest.raises(EmptyNetworkError):
            _simulator(tmp_path).simulate(AlchemicalNetwork())

    def test_raise_disconnected_ligands(self, tmp_path: Path) -> None:
        t_ab = _transformation("1", a="lig_a", b="lig_b")
        t_cd = _transformation("2", a="lig_c", b="lig_d")
        network = AlchemicalNetwork(edges=[t_ab, t_cd])

        with pytest.raises(DisconnectedNetworkError, match="2 disconnected groups"):
            _simulator(tmp_path).simulate(network)

    def test_reject_foreign_protocol(self, tmp_path: Path) -> None:
        other = ForeignProtocol(ForeignProtocol.default_settings())
        network = AlchemicalNetwork(edges=[_transformation("a", protocol=other)])

        with patch("openplaceholder.impl.simulators.execute_DAG") as execute:
            with pytest.raises(UnsupportedProtocolError, match="only RelativeHybridTopologyProtocol is supported"):
                _simulator(tmp_path).simulate(network)

        execute.assert_not_called()
        assert not list(tmp_path.iterdir())

    def test_reject_colliding_names(self, tmp_path: Path) -> None:
        t_ab = _transformation("same", a="lig_a", b="lig_b")
        t_bc = _transformation("same", a="lig_b", b="lig_c")
        network = AlchemicalNetwork(edges=[t_ab, t_bc])

        with patch("openplaceholder.impl.simulators.execute_DAG") as execute:
            with pytest.raises(ValueError, match="transformations do not have unique names"):
                _simulator(tmp_path).simulate(network)

        execute.assert_not_called()

    def test_executes_in_separate_directories(self, tmp_path: Path) -> None:
        network = AlchemicalNetwork(edges=_transformation_chain("abc"))

        with patch("openplaceholder.impl.simulators.execute_DAG", side_effect=_executes()) as execute:
            results = _simulator(tmp_path).simulate(network)

        assert len(results) == 2
        assert {p.name for p in tmp_path.iterdir()} == {"t_a_b", "t_b_c"}
        assert all((tmp_path / e / "dag_result.json").is_file() for e in ("t_a_b", "t_b_c"))
        for call in execute.call_args_list:
            assert call.kwargs["raise_error"] is False
            assert call.kwargs["keep_shared"] is True

    def test_results_have_new_transformation(self, tmp_path: Path) -> None:
        original_transformation = _transformation("edge_a")
        network = AlchemicalNetwork(edges=[original_transformation])

        # determine new run time from defaults
        assert isinstance(original_transformation.protocol.settings, RelativeHybridTopologyProtocolSettings)
        original_length = original_transformation.protocol.settings.simulation_settings.production_length
        assert original_length.magnitude > 0
        new_length = float(original_length.magnitude * 2)

        with patch("openplaceholder.impl.simulators.execute_DAG", side_effect=_executes()):
            results: SimulationResults = _simulator(tmp_path, production_length_ns=new_length).simulate(network)

        new_transformation, _ = next(iter(results))

        assert isinstance(new_transformation.protocol.settings, RelativeHybridTopologyProtocolSettings)
        assert new_transformation.protocol.settings.simulation_settings.production_length == (
            new_length * unit.nanoseconds
        )
        assert original_transformation.key != new_transformation.key

    def test_failed_edge_does_not_block_network(self, tmp_path: Path) -> None:
        transformations = _transformation_chain("abcd")
        network = AlchemicalNetwork(edges=transformations)

        cycler = cycle((False, True))  # generate False, True, False, True, ...

        def execute(dag: ProtocolDAG, **kwargs: object) -> ProtocolDAGResult:
            _ = kwargs
            return _dag_result(dag, ok=next(cycler))

        with patch("openplaceholder.impl.simulators.execute_DAG", side_effect=execute):
            results = _simulator(tmp_path).simulate(network)

        assert len(results) == len(network.edges)
        assert not results.ok()

    def test_failure_without_records_completes(self, tmp_path: Path) -> None:
        network = AlchemicalNetwork(edges=_transformation_chain("abc"))

        with patch(
            "openplaceholder.impl.simulators.execute_DAG",
            side_effect=_executes(ok=False, failures=False),
        ):
            results = _simulator(tmp_path).simulate(network)

        assert len(results) == len(network.edges)

        for _, pdr in results:
            assert isinstance(pdr, ProtocolDAGResult)
            assert not pdr.ok()

    def test_unstartable_edge_skipped(self, tmp_path: Path) -> None:
        """An edge raising outside the protocol units must not end the campaign."""
        transformations = _transformation_chain("abc")
        network = AlchemicalNetwork(edges=transformations)
        calls = cycle([RuntimeError(), None])
        successful = 0

        def execute(dag: ProtocolDAG, **kwargs: object) -> ProtocolDAGResult:
            nonlocal successful
            _ = kwargs
            if isinstance(outcome := next(calls), Exception):
                raise outcome
            successful += 1
            return _dag_result(dag)

        with patch("openplaceholder.impl.simulators.execute_DAG", side_effect=execute):
            results = _simulator(tmp_path).simulate(network)

        # only the edge that ran is recorded
        assert len(results) == successful

        t0_name = transformations[0].name
        t1_name = transformations[1].name

        assert t0_name is not None and t1_name is not None
        assert not (tmp_path / t0_name / "dag_result.json").exists()
        assert (tmp_path / t1_name / "dag_result.json").is_file()

    def test_unserialisable_failure_raises(self, tmp_path: Path) -> None:
        network = AlchemicalNetwork(edges=_transformation_chain("abc"))

        # create a type that the gufe json encoder won't recognize
        class UnknownType: ...

        # return a ProtocolDAGResult with unrecognized outputs
        def make_fake_result(dag: ProtocolDAG, **_: dict[str, Any]) -> ProtocolDAGResult:
            source = dag.protocol_units[0]
            results = [
                ProtocolUnitResult(
                    source_key=source.key,
                    inputs={},
                    outputs={"invalid_data": UnknownType()},
                )
            ]

            return ProtocolDAGResult(
                protocol_units=[source],
                protocol_unit_results=results,
                transformation_key=dag.transformation_key,
            )

        with patch("openplaceholder.impl.simulators.execute_DAG", side_effect=make_fake_result):
            with pytest.raises(TypeError, match=f"{UnknownType.__name__} is not JSON serializable"):
                _simulator(tmp_path).simulate(network)

    def test_simulate_transformation_rejects_foreign_protocol(self, tmp_path: Path) -> None:
        """Running one edge directly must not skip the protocol check."""
        other = ForeignProtocol(ForeignProtocol.default_settings())

        with patch("openplaceholder.impl.simulators.execute_DAG") as execute:
            with pytest.raises(UnsupportedProtocolError, match="RelativeHybridTopologyProtocol"):
                _simulator(tmp_path).simulate_transformation(_transformation("a", protocol=other))

        execute.assert_not_called()
        assert not list(tmp_path.iterdir())
