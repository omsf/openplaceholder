from unittest.mock import patch

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
)
from openplaceholder.tests.helpers import (
    make_network_results,
    make_simulation_results,
)


def default_cinnabar_analyzer() -> CinnabarAnalyzer:
    return CinnabarAnalyzer(SimulationAnalyzerConfig())


class TestExtractFreeEnergies:

    def test_ddg_is_the_complex_leg_less_the_solvent_leg(self) -> None:
        simulation_results = make_simulation_results([("a", "b", -3.0, -1.0)], mbar_error=0.3)
        records = extract_free_energies(simulation_results)

        assert len(records) == 1
        record = records[0]

        assert record[2] == -2.0 * unit.kilocalorie_per_mole
        # the legs' uncertainties add in quadrature
        assert isinstance(uncertainty := record[3], Quantity)
        assert uncertainty.m == pytest.approx((2 * 0.3**2) ** 0.5)

    @pytest.mark.parametrize("repeat_std, per_leg", [(0.0, 0.3), (0.5, 0.5)])
    def test_mbar_error_stands_in_without_a_spread(self, repeat_std: float, per_leg: float) -> None:
        """A single repeat has no spread, so the MBAR error is used instead."""
        simulation_results = make_simulation_results([("a", "b", -3.0, -1.0)], repeat_std=repeat_std, mbar_error=0.3)
        records = extract_free_energies(simulation_results)

        assert len(records) == 1
        record = records[0]

        assert isinstance(uncertainty := record[3], Quantity)
        assert uncertainty.m == pytest.approx((2 * per_leg**2) ** 0.5)

    def test_failed_leg_drops_only_its_own_pair(self) -> None:
        free_energies = extract_free_energies(make_network_results())
        assert len(free_energies) == 6
        # extract ligand pairs from DDGRecords
        pairs = {tuple(fe[0:2]) for fe in free_energies}
        assert ("d", "g") not in pairs


class TestCinnabarAnalyzer:

    def test_estimates_the_largest_component_only(self) -> None:
        analysis = default_cinnabar_analyzer().analyze(make_network_results())

        assert [record[0] for record in analysis.dg] == ["a", "b", "c", "d"]
        assert analysis.statistics["discarded_ligands"] == ["e", "f"]

        # pairs of names
        pairs = [tuple(record[0:2]) for record in analysis.ddg]
        assert ("e", "f") in pairs

    def test_nothing_to_analyze_raises(self) -> None:
        with pytest.raises(NoResultsError):
            default_cinnabar_analyzer().analyze(SimulationResults([]))

        failed = make_simulation_results([("a", "b", -3.0, -1.0)], failed=("a_b_complex", "a_b_solvent"))
        with pytest.raises(NoResultsError, match="both of its legs"):
            default_cinnabar_analyzer().analyze(failed)

    def test_mean_stderr(self) -> None:
        mean, stderr = CinnabarAnalyzer._mean_stderr([0.5 * unit.kilocalorie_per_mole, 0.0 * unit.kilocalorie_per_mole])

        assert mean is not None and mean.m == pytest.approx(0.25)
        assert stderr is not None and stderr.m == pytest.approx(0.25)

    @pytest.mark.parametrize("values", [[], [0.5]])
    def test_stderr_absent_without_a_spread(self, values: list[float]) -> None:
        """numpy would answer NaN here, which JSON cannot represent."""
        assert CinnabarAnalyzer._mean_stderr([value * unit.kilocalorie_per_mole for value in values])[1] is None

    def test_cycle_statistics_absent_without_cycles(self) -> None:
        """A minimal spanning network has no cycles to close."""
        tree = make_simulation_results([("a", "b", -3.0, -1.0), ("b", "c", -2.5, -1.0)])
        statistics = default_cinnabar_analyzer().analyze(tree).statistics

        assert statistics["cycle_closure_error"] is None
        assert statistics["cycle_closure_error_stderr"] is None

    def test_statistics_reported_for_a_cyclic_network(self) -> None:
        statistics = default_cinnabar_analyzer().analyze(make_network_results()).statistics

        assert statistics["cycle_closure_error"].m > 0
        assert statistics["cycle_closure_error_stderr"] is not None
        # plain floats, as every other quantity in the output carries
        assert isinstance(statistics["edge_error"].m, float)

    def test_covariance_follows_the_ligand_order(self) -> None:
        analysis = default_cinnabar_analyzer().analyze(make_network_results())
        order = analysis.statistics["ligand_order"]
        covariance = analysis.statistics["covariance_matrix"]
        reported = {record[0]: record[2].m for record in analysis.dg}

        assert np.allclose(covariance, covariance.T)
        assert np.sqrt(np.diag(covariance)) == pytest.approx([reported[ligand] for ligand in order])


class TestAnalysisResults:

    def test_round_trips_through_json(self) -> None:
        analysis = default_cinnabar_analyzer().analyze(make_network_results())
        content = analysis.to_json()

        with patch.dict(TOKENIZABLE_REGISTRY, {}, clear=True):
            restored = AnalysisResults.from_json(content=content)
            assert restored.ddg == analysis.ddg
            assert restored.dg == analysis.dg

            assert restored is not analysis
            assert [record[2] for record in restored.ddg] == [record[2] for record in analysis.ddg]
            assert restored.statistics["discarded_ligands"] == ["e", "f"]
            assert len(restored.simulation_results) == len(analysis.simulation_results)
