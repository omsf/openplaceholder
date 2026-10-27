import logging
from dataclasses import dataclass
from typing import cast

import networkx as nx
import numpy as np
from cinnabar import FEMap, Measurement
from cinnabar.estimators import MLEEstimatorResult
from openff.units import Quantity, unit

from openplaceholder.core.analysis.analyzer import (
    AnalysisResults,
    Analyzer,
    AnalyzerConfigBase,
    DDGRecord,
    DGRecord,
    NoResultsError,
    extract_free_energies,
)
from openplaceholder.core.simulation.simulator import SimulationResults

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimulationAnalyzerConfig(AnalyzerConfigBase): ...


def _largest_component(ddg: list[DDGRecord]) -> tuple[list[DDGRecord], list[str]]:
    """Network analysis needs a fully connected network. Keep the largest fully connected network."""
    graph = nx.Graph([(fe[0], fe[1]) for fe in ddg])
    # ties break on the sorted names so the choice does not move between runs
    keep = max(nx.connected_components(graph), key=lambda component: (len(component), sorted(component)))

    connected = []

    for fe in ddg:
        if {fe[0], fe[1]}.issubset(keep):
            connected.append(fe)

    return connected, sorted(set(graph.nodes) - keep)


class CinnabarAnalyzer(Analyzer):
    """An Analyzer that estimates free energies with cinnabar."""

    _config: SimulationAnalyzerConfig

    def _setup(self) -> None: ...

    def _analyze(self, results: SimulationResults) -> AnalysisResults:
        free_energies = extract_free_energies(results)
        if len(free_energies) == 0:
            raise NoResultsError("no ligand pair has both of its legs")

        connected, discarded = _largest_component(free_energies)
        if discarded:
            logger.warning("estimating without %s, disconnected from the network", ", ".join(discarded))

        femap = self._femap(connected)

        # collect cycle closure statistics
        closure_df = femap.get_cycle_closure_dataframe()
        closure_deviation: list[Quantity] = []
        for v in closure_df["cc (kcal/mol)"].abs().values:
            closure_deviation.append(float(v) * unit.kilocalories_per_mole)

        cycle_error, cycle_error_stderr = self._mean_stderr(closure_deviation)
        edge_error, edge_error_stderr = self._mean_stderr([record[3] for record in connected])

        dg = self._dg_records(femap)
        # only recorded once the estimates above have been generated
        estimator: MLEEstimatorResult = cast(MLEEstimatorResult, femap.get_estimator_metadata("MLE"))

        statistics = {
            "discarded_ligands": discarded,
            "cycle_closure_error": cycle_error,
            "cycle_closure_error_stderr": cycle_error_stderr,
            "edge_error": edge_error,
            "edge_error_stderr": edge_error_stderr,
            # rows and columns follow ligand_order, so the two travel together
            "covariance_matrix": estimator.covariance_matrix,
            "ligand_order": estimator.ligand_order,
        }

        return AnalysisResults(results, free_energies, dg, statistics)

    @staticmethod
    def _femap(free_energies: list[DDGRecord]) -> FEMap:
        """Load the relative free energies of each ligand pair into a cinnabar ``FEMap``."""
        femap = FEMap()
        for record in free_energies:
            femap.add_measurement(
                Measurement(
                    labelA=record[0],
                    labelB=record[1],
                    DG=record[2],
                    uncertainty=record[3],
                    computational=True,
                    source="calculated",
                )
            )
        return femap

    @staticmethod
    def _dg_records(femap: FEMap) -> list[DGRecord]:
        """Estimate one free energy per ligand from the network as a whole."""
        femap.generate_absolute_values()
        frame = femap.get_absolute_dataframe()
        estimated = frame[frame["computational"] & (frame["source"] == "MLE")]

        return [
            (
                row["label"],
                row["DG (kcal/mol)"] * unit.kilocalorie_per_mole,
                row["uncertainty (kcal/mol)"] * unit.kilocalorie_per_mole,
            )
            for _, row in estimated.sort_values("label").iterrows()
        ]

    @staticmethod
    def _mean_stderr(values: list[Quantity]) -> tuple[Quantity | None, Quantity | None]:
        """Return the arithmetic mean of ``values`` and its estimated standard error."""

        if len(values) == 0:
            return None, None

        magnitudes = np.array([value.m_as(unit.kilocalorie_per_mole) for value in values])
        mean = float(np.mean(magnitudes)) * unit.kilocalorie_per_mole

        if len(magnitudes) == 1:
            return mean, None

        stderr = float(np.std(magnitudes, ddof=1)) / len(magnitudes) ** 0.5
        return mean, stderr * unit.kilocalorie_per_mole
