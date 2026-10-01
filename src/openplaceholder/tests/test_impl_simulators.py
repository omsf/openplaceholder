from pathlib import Path
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
from gufe.protocols import ProtocolDAG, ProtocolDAGResult
from gufe.protocols.protocolunit import ProtocolUnitFailure, ProtocolUnitResult
from openfe.protocols.openmm_afe import AbsoluteSolvationProtocol
from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
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


def _ligand(name: str, smiles: str = "c1ccccc1") -> SmallMoleculeComponent:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    EmbedMolecule(mol, randomSeed=0xF00D)
    return SmallMoleculeComponent(mol, name=name)


def _transformation(name: str, a: str = "lig_a", b: str = "lig_b", protocol: object = PROTOCOL) -> Transformation:
    """A solvent-leg RBFE transformation between two named ligands."""
    ligand_a, ligand_b = _ligand(a), _ligand(b)
    return Transformation(
        stateA=ChemicalSystem({"ligand": ligand_a, "solvent": SolventComponent()}),
        stateB=ChemicalSystem({"ligand": ligand_b, "solvent": SolventComponent()}),
        protocol=protocol,
        mapping=LigandAtomMapping(ligand_a, ligand_b, {i: i for i in range(ligand_a.to_rdkit().GetNumAtoms())}),
        name=name,
    )


def _dag_result(dag: ProtocolDAG, ok: bool = True, failures: bool = True) -> ProtocolDAGResult:
    """A result for ``dag``, keyed to it exactly as a real execution would be."""
    source = dag.protocol_units[0]
    if ok:
        results: list[ProtocolUnitResult] = [ProtocolUnitResult(source_key=source.key, inputs={}, outputs={})]
    elif failures:
        results = [
            ProtocolUnitFailure(
                source_key=source.key,
                inputs={},
                outputs={},
                exception=("RuntimeError", ("boom",)),
                traceback="tb",
            )
        ]
    else:
        results = []
    return ProtocolDAGResult(
        protocol_units=[source],
        protocol_unit_results=results,
        transformation_key=dag.transformation_key,
    )


def _executes(ok: bool = True, failures: bool = True) -> object:
    """Stand in for execute_DAG."""
    return lambda dag, **kwargs: _dag_result(dag, ok=ok, failures=failures)


def _simulator(tmp_path: Path, production_length_ns: float = 5.0) -> OpenFESimulator:
    return OpenFESimulator(
        OpenFESimulatorConfig(simulation_directory=tmp_path, production_length_ns=production_length_ns)
    )


class TestSimulationResults:

    def test_pairs_each_result_with_the_transformation_that_produced_it(self) -> None:
        a, b = _transformation("a", b="lig_b"), _transformation("b", a="lig_b", b="lig_c")
        results = [_dag_result(b.create()), _dag_result(a.create())]  # deliberately out of order

        paired = list(SimulationResults(AlchemicalNetwork(edges=[a, b]), results))

        assert [transformation.name for transformation, _ in paired] == ["b", "a"]

    def test_unmatched_result_raises(self) -> None:
        a, orphan = _transformation("a"), _transformation("orphan", a="lig_x", b="lig_y")
        results = SimulationResults(AlchemicalNetwork(edges=[a]), [_dag_result(orphan.create())])

        with pytest.raises(KeyError, match="no transformation"):
            list(results)

    def test_ok_requires_results(self) -> None:
        a = _transformation("a")

        assert SimulationResults(AlchemicalNetwork(edges=[a]), [_dag_result(a.create())]).ok()
        assert not SimulationResults(AlchemicalNetwork(), []).ok()
        assert not SimulationResults(AlchemicalNetwork(edges=[a]), [_dag_result(a.create(), ok=False)]).ok()

    def test_round_trips_through_json(self) -> None:
        a = _transformation("a")
        results = SimulationResults(AlchemicalNetwork(edges=[a]), [_dag_result(a.create())])

        restored = SimulationResults.from_json(content=results.to_json())

        assert len(restored) == len(results)
        assert [t.name for t, _ in restored] == [t.name for t, _ in results]


class TestOpenFESimulator:

    def test_init(self, tmp_path: Path) -> None:
        _simulator(tmp_path)

    def test_settings_follow_asap_defaults(self, tmp_path: Path) -> None:
        settings = _simulator(tmp_path, production_length_ns=0.5)._protocol.settings
        defaults = RelativeHybridTopologyProtocol.default_settings()

        assert settings.simulation_settings.production_length == 0.5 * unit.nanoseconds
        assert settings.forcefield_settings != defaults.forcefield_settings
        assert settings.protocol_repeats != defaults.protocol_repeats

    def test_empty_network_raises(self, tmp_path: Path) -> None:
        with pytest.raises(EmptyNetworkError):
            _simulator(tmp_path).simulate(AlchemicalNetwork())

    def test_disconnected_ligands_raise(self, tmp_path: Path) -> None:
        network = AlchemicalNetwork(edges=[_transformation("a"), _transformation("b", a="lig_c", b="lig_d")])

        with pytest.raises(DisconnectedNetworkError, match="2 disconnected groups"):
            _simulator(tmp_path).simulate(network)

    def test_foreign_protocol_is_rejected_before_any_work(self, tmp_path: Path) -> None:
        other = AbsoluteSolvationProtocol(settings=AbsoluteSolvationProtocol.default_settings())
        network = AlchemicalNetwork(edges=[_transformation("a", protocol=other)])

        with patch("openplaceholder.impl.simulators.execute_DAG") as execute:
            with pytest.raises(UnsupportedProtocolError, match="RelativeHybridTopologyProtocol"):
                _simulator(tmp_path).simulate(network)

        execute.assert_not_called()
        assert not list(tmp_path.iterdir())

    def test_colliding_names_are_rejected_before_any_work(self, tmp_path: Path) -> None:
        network = AlchemicalNetwork(edges=[_transformation("same"), _transformation("same", a="lig_b", b="lig_c")])

        with patch("openplaceholder.impl.simulators.execute_DAG") as execute:
            with pytest.raises(ValueError, match="unique names"):
                _simulator(tmp_path).simulate(network)

        execute.assert_not_called()

    def test_runs_every_edge_into_its_own_directory(self, tmp_path: Path) -> None:
        network = AlchemicalNetwork(edges=[_transformation("edge_a"), _transformation("edge_b", a="lig_b", b="lig_c")])

        with patch("openplaceholder.impl.simulators.execute_DAG", side_effect=_executes()) as execute:
            results = _simulator(tmp_path).simulate(network)

        assert len(results) == 2
        assert {p.name for p in tmp_path.iterdir()} == {"edge_a", "edge_b"}
        assert all((tmp_path / e / "dag_result.json").is_file() for e in ("edge_a", "edge_b"))
        for call in execute.call_args_list:
            assert call.kwargs["raise_error"] is False
            assert call.kwargs["keep_shared"] is True

    def test_results_pair_back_to_the_transformations_as_run(self, tmp_path: Path) -> None:
        """The protocol is swapped per run, so results must key to that, not the input."""
        network = AlchemicalNetwork(edges=[_transformation("edge_a")])

        with patch("openplaceholder.impl.simulators.execute_DAG", side_effect=_executes()):
            results = _simulator(tmp_path, production_length_ns=0.5).simulate(network)

        transformation, _ = next(iter(results))
        assert transformation.protocol.settings.simulation_settings.production_length == (0.5 * unit.nanoseconds)

    def test_failed_edge_does_not_stop_the_network(self, tmp_path: Path) -> None:
        network = AlchemicalNetwork(edges=[_transformation("edge_a"), _transformation("edge_b", a="lig_b", b="lig_c")])
        outcomes = iter([False, True])

        def execute(dag: ProtocolDAG, **kwargs: object) -> ProtocolDAGResult:
            return _dag_result(dag, ok=next(outcomes))

        with patch("openplaceholder.impl.simulators.execute_DAG", side_effect=execute):
            results = _simulator(tmp_path).simulate(network)

        assert len(results) == 2
        assert not results.ok()

    def test_failure_without_recorded_failures_still_completes(self, tmp_path: Path) -> None:
        network = AlchemicalNetwork(edges=[_transformation("edge_a")])

        with patch(
            "openplaceholder.impl.simulators.execute_DAG",
            side_effect=_executes(ok=False, failures=False),
        ):
            results = _simulator(tmp_path).simulate(network)

        assert len(results) == 1

    def test_edge_that_cannot_start_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        """An edge raising outside the protocol units must not end the campaign."""
        network = AlchemicalNetwork(edges=[_transformation("edge_a"), _transformation("edge_b", a="lig_b", b="lig_c")])
        calls = iter([ValueError("A single LigandAtomMapping is expected"), None])

        def execute(dag: ProtocolDAG, **kwargs: object) -> ProtocolDAGResult:
            if isinstance(outcome := next(calls), Exception):
                raise outcome
            return _dag_result(dag)

        with patch("openplaceholder.impl.simulators.execute_DAG", side_effect=execute):
            results = _simulator(tmp_path).simulate(network)

        # only the edge that ran is recorded, and network/results stay 1:1
        assert len(results) == 1
        assert len(results.network.edges) == 1
        assert not (tmp_path / "edge_a" / "dag_result.json").exists()
        assert (tmp_path / "edge_b" / "dag_result.json").is_file()

    def test_unserialisable_failure_does_not_end_the_run(self, tmp_path: Path) -> None:
        """A result that cannot be written must cost one file, not the run."""
        network = AlchemicalNetwork(edges=[_transformation("edge_a"), _transformation("edge_b", a="lig_b", b="lig_c")])

        with (
            patch("openplaceholder.impl.simulators.execute_DAG", side_effect=_executes()),
            patch.object(ProtocolDAGResult, "to_json", side_effect=TypeError("not serializable")),
        ):
            results = _simulator(tmp_path).simulate(network)

        assert len(results) == 2
        assert not any(tmp_path.glob("*/dag_result.json"))

    def test_simulate_transformation_runs_one_edge(self, tmp_path: Path) -> None:
        transformation = _transformation("edge_a")

        with patch("openplaceholder.impl.simulators.execute_DAG", side_effect=_executes()):
            ran = _simulator(tmp_path).simulate_transformation(transformation)

        assert ran is not None
        executed, result = ran
        assert result.transformation_key == executed.key
        assert (tmp_path / "edge_a" / "dag_result.json").is_file()

    def test_simulate_transformation_returns_none_when_it_cannot_start(self, tmp_path: Path) -> None:
        with patch("openplaceholder.impl.simulators.execute_DAG", side_effect=OSError("disk full")):
            assert _simulator(tmp_path).simulate_transformation(_transformation("edge_a")) is None
