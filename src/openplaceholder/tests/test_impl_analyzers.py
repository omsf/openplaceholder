from typing import Any

import numpy as np
import pytest
from gufe.tokenization import TOKENIZABLE_REGISTRY
from openff.units import Quantity, unit

from openplaceholder.core.analysis.analyzer import (
    AnalysisResults,
    NoResultsError,
    extract_free_energies,
)
from openplaceholder.core.simulation.simulator import SimulationResults
from openplaceholder.impl.analyzers import (
    CinnabarAnalyzer,
    SimulationAnalyzerConfig,
    _cycle_closure_errors,
    _mean_stderr,
)
from openplaceholder.tests.helpers import (
    make_network_results,
    make_simulation_results,
)


def _analyzer() -> CinnabarAnalyzer:
    return CinnabarAnalyzer(SimulationAnalyzerConfig())


def _pairs(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(record["stateA"], record["stateB"]): record for record in records}


class TestExtractFreeEnergies:

    def test_ddg_is_the_complex_leg_less_the_solvent_leg(self) -> None:
        simulation_results = make_simulation_results([("a", "b", -3.0, -1.0)], mbar_error=0.3)
        records = extract_free_energies(simulation_results)

        assert len(records) == 1
        record = records[0]

        assert record["ddg"] == -2.0 * unit.kilocalorie_per_mole
        # the legs' uncertainties add in quadrature
        assert isinstance(uncertainty := record["uncertainty"], Quantity)
        assert uncertainty.m == pytest.approx((2 * 0.3**2) ** 0.5)

    @pytest.mark.parametrize("repeat_std, per_leg", [(0.0, 0.3), (0.5, 0.5)])
    def test_mbar_error_stands_in_without_a_spread(self, repeat_std: float, per_leg: float) -> None:
        """A single repeat has no spread, so the MBAR error is used instead."""
        simulation_results = make_simulation_results([("a", "b", -3.0, -1.0)], repeat_std=repeat_std, mbar_error=0.3)
        records = extract_free_energies(simulation_results)

        assert len(records) == 1
        record = records[0]

        assert isinstance(uncertainty := record["uncertainty"], Quantity)
        assert uncertainty.m == pytest.approx((2 * per_leg**2) ** 0.5)

    def test_failed_leg_drops_only_its_own_pair(self) -> None:
        pairs = _pairs(extract_free_energies(make_network_results()))

        assert ("d", "g") not in pairs
        assert len(pairs) == 6


class TestCinnabarAnalyzer:

    def test_estimates_the_largest_component_only(self) -> None:
        analysis = _analyzer().analyze(make_network_results())

        assert [record["ligand"] for record in analysis.dg] == ["a", "b", "c", "d"]
        assert analysis.statistics["discarded_ligands"] == ["e", "f"]
        # the pair remains a measurement, it just cannot share one offset
        assert ("e", "f") in _pairs(analysis.ddg)

    def test_nothing_to_analyze_raises(self) -> None:
        with pytest.raises(NoResultsError):
            _analyzer().analyze(SimulationResults([]))

        failed = make_simulation_results([("a", "b", -3.0, -1.0)], failed=("a_b_complex", "a_b_solvent"))
        with pytest.raises(NoResultsError, match="both of its legs"):
            _analyzer().analyze(failed)


class TestStatistics:

    def test_mean_stderr(self) -> None:
        mean, stderr = _mean_stderr([0.5 * unit.kilocalorie_per_mole, 0.0 * unit.kilocalorie_per_mole])

        assert mean is not None and mean.m == pytest.approx(0.25)
        assert stderr is not None and stderr.m == pytest.approx(0.25)

    def test_cycle_closure_errors_are_magnitudes(self) -> None:
        """A signed closure must never reach the mean, where it would cancel."""
        assert [error.m for error in _cycle_closure_errors([{"closure": -0.5 * unit.kilocalorie_per_mole}])] == [0.5]

    @pytest.mark.parametrize("values", [[], [0.5]])
    def test_stderr_absent_without_a_spread(self, values: list[float]) -> None:
        """numpy would answer NaN here, which JSON cannot represent."""
        assert _mean_stderr([value * unit.kilocalorie_per_mole for value in values])[1] is None

    def test_cycle_statistics_absent_without_cycles(self) -> None:
        """A minimal spanning network has no cycles to close."""
        tree = make_simulation_results([("a", "b", -3.0, -1.0), ("b", "c", -2.5, -1.0)])
        statistics = _analyzer().analyze(tree).statistics

        assert statistics["cycle_closure_error"] is None
        assert statistics["cycle_closure_error_stderr"] is None

    def test_statistics_reported_for_a_cyclic_network(self) -> None:
        statistics = _analyzer().analyze(make_network_results()).statistics

        assert statistics["cycle_closure_error"].m > 0
        assert statistics["cycle_closure_error_stderr"] is not None
        # plain floats, as every other quantity in the output carries
        assert isinstance(statistics["edge_error"].m, float)

    def test_covariance_follows_the_ligand_order(self) -> None:
        analysis = _analyzer().analyze(make_network_results())
        order = analysis.statistics["ligand_order"]
        covariance = analysis.statistics["covariance_matrix"]
        reported = {record["ligand"]: record["uncertainty"].m for record in analysis.dg}

        assert np.allclose(covariance, covariance.T)
        assert np.sqrt(np.diag(covariance)) == pytest.approx([reported[ligand] for ligand in order])


class TestAnalysisResults:

    def test_round_trips_through_json(self) -> None:
        analysis = _analyzer().analyze(make_network_results())
        content = analysis.to_json()

        # gufe returns the same object from its registry, so clear it to make
        # the restore go through the serialized form rather than the cache
        TOKENIZABLE_REGISTRY.clear()
        restored = AnalysisResults.from_json(content=content)

        assert restored is not analysis
        assert [record["ddg"] for record in restored.ddg] == [record["ddg"] for record in analysis.ddg]
        assert restored.statistics["discarded_ligands"] == ["e", "f"]
        assert len(restored.simulation_results) == len(analysis.simulation_results)
