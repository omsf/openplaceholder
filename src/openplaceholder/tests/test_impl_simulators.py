from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from gufe import SmallMoleculeComponent
from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
from openff.units import unit
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule

from openplaceholder.core.simulation.simulator import (
    DisconnectedNetworkError,
    EmptyNetworkError,
    UnsupportedProtocolError,
)
from openplaceholder.impl.simulators import OpenFESimulator, OpenFESimulatorConfig


class _FakeDAGResult:
    def __init__(self, ok: bool = True) -> None:
        self._ok = ok
        self.protocol_unit_failures = [] if ok else [type("F", (), {"exception": "boom"})()]

    def ok(self) -> bool:
        return self._ok

    def to_json(self, path: Path) -> None:
        Path(path).write_text("{}")


def _ligand(name: str) -> SmallMoleculeComponent:
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    EmbedMolecule(mol, randomSeed=0xF00D)
    return SmallMoleculeComponent(mol, name=name)


class _FakeSystem:
    """Stands in for a ChemicalSystem; only ``components`` is read."""

    def __init__(self, ligand: str) -> None:
        self.components = {"ligand": _ligand(ligand)}


#: a real protocol instance, so _validate sees a supported one on the fakes
_RHT = RelativeHybridTopologyProtocol(settings=RelativeHybridTopologyProtocol.default_settings())


class _FakeTransformation:
    def __init__(self, name: str, state_a: str = "lig_a", state_b: str = "lig_b") -> None:
        self.protocol = _RHT
        self.name = name
        self.key = f"key-{name}"
        self.stateA = _FakeSystem(state_a)
        self.stateB = _FakeSystem(state_b)
        self.mapping = None


class _ResultsStub(list):  # type: ignore[type-arg]
    """Stands in for SimulationResults, which tokenizes its contents on init."""

    def __init__(self, network: object, dag_results: list) -> None:  # type: ignore[type-arg]
        super().__init__(dag_results)
        self.network = network


class _FakeNetwork:
    def __init__(self, names: list[str], edges: list[tuple[str, str]] | None = None) -> None:
        pairs = edges or [("lig_a", "lig_b")] * len(names)
        self.edges = [_FakeTransformation(n, a, b) for n, (a, b) in zip(names, pairs)]


class TestSimulationResults:

    def test_pairs_results_with_the_transformations_that_produced_them(self) -> None:
        from openplaceholder.core.simulation.simulator import SimulationResults

        class _T:
            def __init__(self, key: str) -> None:
                self.key = key

        class _Net:
            def __init__(self, edges: list[object]) -> None:
                self.edges = edges

        class _R:
            def __init__(self, key: str) -> None:
                self.transformation_key = key

        a, b = _T("k-a"), _T("k-b")
        # deliberately out of order: the join must be by key, not position
        results = SimulationResults.__new__(SimulationResults)
        results.network, results.dag_results = _Net([a, b]), [_R("k-b"), _R("k-a")]

        assert [t for t, _ in results] == [b, a]

    def test_empty_results_are_not_ok(self) -> None:
        """A run where every edge was skipped must not report success."""
        from openplaceholder.core.simulation.simulator import SimulationResults

        class _Net:
            edges: list[object] = []

        results = SimulationResults.__new__(SimulationResults)
        results.network, results.dag_results = _Net(), []

        assert len(results) == 0
        assert results.ok() is False

    def test_unmatched_result_is_an_error_not_a_silent_skip(self) -> None:
        from openplaceholder.core.simulation.simulator import SimulationResults

        class _Net:
            edges: list[object] = []

        class _R:
            transformation_key = "k-missing"

        results = SimulationResults.__new__(SimulationResults)
        results.network, results.dag_results = _Net(), [_R()]

        with pytest.raises(KeyError, match="k-missing"):
            list(results)


class TestOpenFESimulator:

    def test_init(self, tmp_path: Path) -> None:
        OpenFESimulator(OpenFESimulatorConfig(simulation_directory=tmp_path))

    def test_settings_follow_asap_defaults(self, tmp_path: Path) -> None:
        simulator = OpenFESimulator(
            OpenFESimulatorConfig(simulation_directory=tmp_path, production_length_ns=0.5, equilibration_length_ns=0.2)
        )
        settings = simulator._protocol.settings

        assert settings.simulation_settings.production_length == 0.5 * unit.nanoseconds
        assert settings.simulation_settings.equilibration_length == 0.2 * unit.nanoseconds
        assert settings.forcefield_settings.small_molecule_forcefield == "openff-2.2.0.offxml"
        assert settings.solvation_settings.box_shape == "dodecahedron"
        assert settings.alchemical_settings.softcore_LJ == "gapsys"
        assert settings.alchemical_settings.turn_off_core_unique_exceptions is False
        assert settings.thermo_settings.temperature == 298.15 * unit.kelvin
        assert settings.protocol_repeats == 1

    def test_empty_network_raises(self, tmp_path: Path) -> None:
        simulator = OpenFESimulator(OpenFESimulatorConfig(simulation_directory=tmp_path))

        with pytest.raises(EmptyNetworkError):
            simulator.simulate(_FakeNetwork([]))

    def test_runs_every_edge_into_its_own_directory(self, tmp_path: Path) -> None:
        simulator = OpenFESimulator(OpenFESimulatorConfig(simulation_directory=tmp_path))
        network = _FakeNetwork(["edge_a", "edge_b"])

        with (
            patch.object(OpenFESimulator, "_rebuild") as rebuild,
            patch("openplaceholder.impl.simulators.execute_DAG") as execute,
            # SimulationResults tokenizes its contents on construction, so stand it
            # in with a plain list to keep this test about the execution loop
            patch("openplaceholder.impl.simulators.SimulationResults", _ResultsStub),
            patch("openplaceholder.impl.simulators.AlchemicalNetwork", lambda edges: list(edges)),
        ):
            execute.return_value = _FakeDAGResult()
            simulator._protocol.gather = lambda _: type("R", (), {"get_estimate": lambda self: 1.0})()
            results = simulator.simulate(network)

        assert len(results) == 2
        assert rebuild.call_count == 2
        assert {p.name for p in tmp_path.iterdir()} == {"edge_a", "edge_b"}
        # each edge is persisted as it completes, not only at the end
        assert all((tmp_path / e / "dag_result.json").is_file() for e in ("edge_a", "edge_b"))
        for call in execute.call_args_list:
            assert call.kwargs["raise_error"] is False
            assert call.kwargs["keep_shared"] is True

    def test_failed_edge_does_not_stop_the_network(self, tmp_path: Path) -> None:
        simulator = OpenFESimulator(OpenFESimulatorConfig(simulation_directory=tmp_path))
        network = _FakeNetwork(["edge_a", "edge_b"])

        with (
            patch.object(OpenFESimulator, "_rebuild"),
            patch("openplaceholder.impl.simulators.execute_DAG") as execute,
            patch("openplaceholder.impl.simulators.SimulationResults", _ResultsStub),
            patch("openplaceholder.impl.simulators.AlchemicalNetwork", lambda edges: list(edges)),
        ):
            execute.side_effect = [_FakeDAGResult(ok=False), _FakeDAGResult(ok=True)]
            simulator._protocol.gather = lambda _: type("R", (), {"get_estimate": lambda self: 1.0})()
            results: Any = simulator.simulate(network)

        # the failing edge is recorded rather than aborting the remaining edges
        assert len(results) == 2
        assert [r.ok() for r in results] == [False, True]

    def test_disconnected_ligands_raise(self, tmp_path: Path) -> None:
        simulator = OpenFESimulator(OpenFESimulatorConfig(simulation_directory=tmp_path))
        # a->b and c->d never meet, so their dG estimates share no reference
        network = _FakeNetwork(["e1", "e2"], edges=[("lig_a", "lig_b"), ("lig_c", "lig_d")])

        with pytest.raises(DisconnectedNetworkError, match="2 disconnected groups"):
            simulator.simulate(network)

    def test_rbfe_legs_are_not_mistaken_for_a_split(self, tmp_path: Path) -> None:
        """Complex and solvent legs touch disjoint ChemicalSystems but the same ligands."""
        simulator = OpenFESimulator(OpenFESimulatorConfig(simulation_directory=tmp_path))
        network = _FakeNetwork(
            ["a_complex_b", "a_solvent_b", "b_complex_c", "b_solvent_c"],
            edges=[("lig_a", "lig_b"), ("lig_a", "lig_b"), ("lig_b", "lig_c"), ("lig_b", "lig_c")],
        )

        with (
            patch.object(OpenFESimulator, "_rebuild"),
            patch("openplaceholder.impl.simulators.execute_DAG") as execute,
            patch("openplaceholder.impl.simulators.SimulationResults", _ResultsStub),
            patch("openplaceholder.impl.simulators.AlchemicalNetwork", lambda edges: list(edges)),
        ):
            execute.return_value = _FakeDAGResult()
            simulator._protocol.gather = lambda _: type("R", (), {"get_estimate": lambda self: 1.0})()
            results = simulator.simulate(network)

        assert len(results) == 4

    def test_unserialisable_failure_does_not_end_the_run(self, tmp_path: Path) -> None:
        """One edge whose result cannot be written must not abort the network."""
        simulator = OpenFESimulator(OpenFESimulatorConfig(simulation_directory=tmp_path))
        network = _FakeNetwork(["edge_a", "edge_b"])

        class _UnserialisableDAGResult(_FakeDAGResult):
            def to_json(self, path: Path) -> None:
                raise TypeError("Object of type object is not JSON serializable")

        bad = _UnserialisableDAGResult(ok=False)

        with (
            patch.object(OpenFESimulator, "_rebuild"),
            patch("openplaceholder.impl.simulators.execute_DAG") as execute,
            patch("openplaceholder.impl.simulators.SimulationResults", _ResultsStub),
            patch("openplaceholder.impl.simulators.AlchemicalNetwork", lambda edges: list(edges)),
        ):
            execute.side_effect = [bad, _FakeDAGResult(ok=True)]
            simulator._protocol.gather = lambda _: type("R", (), {"get_estimate": lambda self: 1.0})()
            results = simulator.simulate(network)

        assert len(results) == 2
        assert not (tmp_path / "edge_a" / "dag_result.json").exists()
        assert (tmp_path / "edge_b" / "dag_result.json").is_file()

    def test_failure_without_recorded_failures_still_logs(self, tmp_path: Path) -> None:
        simulator = OpenFESimulator(OpenFESimulatorConfig(simulation_directory=tmp_path))
        empty = _FakeDAGResult(ok=False)
        empty.protocol_unit_failures = []

        with (
            patch.object(OpenFESimulator, "_rebuild"),
            patch("openplaceholder.impl.simulators.execute_DAG") as execute,
            patch("openplaceholder.impl.simulators.SimulationResults", _ResultsStub),
            patch("openplaceholder.impl.simulators.AlchemicalNetwork", lambda edges: list(edges)),
        ):
            execute.return_value = empty
            results = simulator.simulate(_FakeNetwork(["edge_a"]))

        assert len(results) == 1

    def test_foreign_protocol_is_rejected_before_any_work(self, tmp_path: Path) -> None:
        """_rebuild would silently swap the protocol, so refuse up front."""
        simulator = OpenFESimulator(OpenFESimulatorConfig(simulation_directory=tmp_path))
        network = _FakeNetwork(["edge_a"])
        network.edges[0].protocol = object()  # not a RelativeHybridTopologyProtocol

        with patch("openplaceholder.impl.simulators.execute_DAG") as execute:
            with pytest.raises(UnsupportedProtocolError, match="RelativeHybridTopologyProtocol"):
                simulator.simulate(network)

        execute.assert_not_called()
        assert not list(tmp_path.iterdir())

    def test_colliding_names_are_rejected_before_any_work(self, tmp_path: Path) -> None:
        """Two edges sharing a name would share a directory and clobber each other."""
        simulator = OpenFESimulator(OpenFESimulatorConfig(simulation_directory=tmp_path))
        network = _FakeNetwork(["same", "same"], edges=[("lig_a", "lig_b"), ("lig_b", "lig_c")])

        with patch("openplaceholder.impl.simulators.execute_DAG") as execute:
            with pytest.raises(ValueError, match="unique names"):
                simulator.simulate(network)

        execute.assert_not_called()

    def test_edge_that_cannot_start_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        """create() raising (e.g. no atom mapping) must not end the campaign."""
        simulator = OpenFESimulator(OpenFESimulatorConfig(simulation_directory=tmp_path))
        network = _FakeNetwork(["edge_a", "edge_b"])

        with (
            patch.object(OpenFESimulator, "_rebuild"),
            patch("openplaceholder.impl.simulators.execute_DAG") as execute,
            patch("openplaceholder.impl.simulators.SimulationResults", _ResultsStub),
            patch("openplaceholder.impl.simulators.AlchemicalNetwork", lambda edges: list(edges)),
        ):
            execute.side_effect = [ValueError("A single LigandAtomMapping is expected"), _FakeDAGResult()]
            simulator._protocol.gather = lambda _: type("R", (), {"get_estimate": lambda self: 1.0})()
            results = simulator.simulate(network)

        # only the edge that ran is recorded, and network/results stay 1:1
        assert len(results) == 1
        assert len(results.network) == 1
        assert not (tmp_path / "edge_a" / "dag_result.json").exists()
        assert (tmp_path / "edge_b" / "dag_result.json").is_file()
