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

    def test_ddg_equals_complex_minus_solvent(self) -> None:
        simulation_results = make_simulation_results([("a", "b", -3.0, -1.0)], mbar_error=0.3)
        records, _ = extract_free_energies(simulation_results)

        # we expect only 1 record
        assert len(records) == 1
        record = records[0]

        assert record[2] == -2.0 * unit.kilocalorie_per_mole
        assert isinstance(uncertainty := record[3], Quantity)
        assert uncertainty.m == pytest.approx((2 * 0.3**2) ** 0.5)

    @pytest.mark.parametrize("repeat_std, per_leg", [(0.0, 0.3), (0.5, 0.5)])
    def test_mbar_error_fallback(self, repeat_std: float, per_leg: float) -> None:
        """A single repeat has no spread, so the MBAR error is used."""
        simulation_results = make_simulation_results([("a", "b", -3.0, -1.0)], repeat_std=repeat_std, mbar_error=0.3)
        records, _ = extract_free_energies(simulation_results)

        assert len(records) == 1
        record = records[0]

        assert isinstance(uncertainty := record[3], Quantity)
        assert uncertainty.m == pytest.approx((2 * per_leg**2) ** 0.5)

    def test_failed_leg_drops_own_pair(self) -> None:
        free_energies, dropped = extract_free_energies(make_network_results())
        assert len(free_energies) == 7
        # extract ligand pairs from DDGRecords
        assert dropped == {"i"}


class TestCinnabarAnalyzer:

    def test_analyzer_name(self) -> None:
        analysis = default_cinnabar_analyzer().analyze(make_network_results())
        assert analysis.analyzer_name == CinnabarAnalyzer.__name__

    def test_estimates_the_largest_component_only(self) -> None:
        analysis = default_cinnabar_analyzer().analyze(make_network_results())

        assert analysis.ligands() == {"a", "b", "c", "d"}
        assert set(analysis.meta["statistics"]["discarded_ligands"]) == set(["e", "f", "g", "h", "i"])
        assert ("e", "f") in analysis.ddg_pairs()

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
        statistics = default_cinnabar_analyzer().analyze(tree).meta["statistics"]

        assert statistics["cycle_closure_error"] is None
        assert statistics["cycle_closure_error_stderr"] is None

    def test_statistics_reported_for_a_cyclic_network(self) -> None:
        statistics = default_cinnabar_analyzer().analyze(make_network_results()).meta["statistics"]

        assert statistics["cycle_closure_error"].m > 0
        assert statistics["cycle_closure_error_stderr"] is not None
        # plain floats, as every other quantity in the output carries
        assert isinstance(statistics["edge_error"].m, float)

    def test_covariance_follows_the_ligand_order(self) -> None:
        analysis = default_cinnabar_analyzer().analyze(make_network_results())
        order = analysis.meta["statistics"]["ligand_order"]
        covariance = analysis.meta["statistics"]["covariance_matrix"]

        reported = {}
        for lig in analysis.ligands():
            if (q := analysis.dg(lig)) is not None:
                reported[lig] = q[1].m

        assert np.allclose(covariance, covariance.T)
        assert np.sqrt(np.diag(covariance)) == pytest.approx([reported[ligand] for ligand in order])


class TestAnalysisResults:

    def test_round_trips_through_json(self) -> None:
        analysis = default_cinnabar_analyzer().analyze(make_network_results())
        content = analysis.to_json()

        with patch.dict(TOKENIZABLE_REGISTRY, {}, clear=True):
            restored = AnalysisResults.from_json(content=content)
            assert restored._ddg == analysis._ddg
            assert restored._dg == analysis._dg
            assert restored.key == analysis.key
            assert restored is not analysis
