import logging
from dataclasses import dataclass
from typing import Any, cast

import networkx as nx
import numpy as np
from cinnabar import FEMap, Measurement
from cinnabar.estimators import MLEEstimatorResult
from openff.units import unit
from pint import Quantity

from openplaceholder.core.analysis.analyzer import (
    AnalysisResults,
    Analyzer,
    AnalyzerConfigBase,
    NoResultsError,
    extract_free_energies,
)
from openplaceholder.core.simulation.simulator import SimulationResults

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimulationAnalyzerConfig(AnalyzerConfigBase): ...


def _largest_component(ddg: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Network analysis needs a fully connected network. Keep the largest fully connected network."""
    graph = nx.Graph([(record["stateA"], record["stateB"]) for record in ddg])
    # ties break on the sorted names so the choice does not move between runs
    keep = max(nx.connected_components(graph), key=lambda component: (len(component), sorted(component)))

    connected = []

    for record in ddg:
        if {record["stateA"], record["stateB"]}.issubset(keep):
            connected.append(record)

    return connected, sorted(set(graph.nodes) - keep)


def _femap(free_energies: list[dict[str, Any]]) -> FEMap:
    """Load the relative free energies of each ligand pair into a cinnabar ``FEMap``."""
    femap = FEMap()
    for record in free_energies:
        femap.add_measurement(
            Measurement(
                labelA=record["stateA"],
                labelB=record["stateB"],
                DG=record["ddg"],
                uncertainty=record["uncertainty"],
                computational=True,
                source="calculated",
            )
        )
    return femap


def _dg_records(femap: FEMap) -> list[dict[str, Any]]:
    """Estimate one free energy per ligand from the network as a whole."""
    femap.generate_absolute_values()
    frame = femap.get_absolute_dataframe()
    estimated = frame[frame["computational"] & (frame["source"] == "MLE")]

    return [
        {
            "ligand": row["label"],
            "dg": row["DG (kcal/mol)"] * unit.kilocalorie_per_mole,
            "uncertainty": row["uncertainty (kcal/mol)"] * unit.kilocalorie_per_mole,
        }
        for _, row in estimated.sort_values("label").iterrows()
    ]


def _mean_stderr(values: list[Quantity]) -> tuple[Quantity | None, Quantity | None]:
    """The average of ``values``, and the standard error of that average."""
    magnitudes = np.array([value.m_as(unit.kilocalorie_per_mole) for value in values])
    if len(magnitudes) == 0:
        return None, None

    mean = float(np.mean(magnitudes)) * unit.kilocalorie_per_mole
    if len(magnitudes) == 1:
        return mean, None

    stderr = float(np.std(magnitudes, ddof=1)) / len(magnitudes) ** 0.5
    return mean, stderr * unit.kilocalorie_per_mole


def _cycle_closures(femap: FEMap) -> list[dict[str, Any]]:
    """Every cycle in the network, with how far it is from summing to zero."""
    frame = femap.get_cycle_closure_dataframe()
    return [
        {
            # a list, as the tuple cinnabar hands back returns from JSON as one
            "cycle": list(row["cycle"]),
            "closure": row["cc (kcal/mol)"] * unit.kilocalorie_per_mole,
            "closure_per_edge": row["cc_per_edge (kcal/mol)"] * unit.kilocalorie_per_mole,
            "closure_over_uncertainty": row["cc_unc_normalized"],
        }
        for _, row in frame.iterrows()
    ]


def _cycle_closure_errors(cycles: list[dict[str, Any]]) -> list[Quantity]:
    """How far each cycle is from closing, as a magnitude."""
    return [abs(cycle["closure"]) for cycle in cycles]


class CinnabarAnalyzer(Analyzer):
    """An Analyzer that estimates free energies with cinnabar."""

    _config: SimulationAnalyzerConfig

    def _setup(self) -> None: ...

    def _analyze(self, results: SimulationResults) -> AnalysisResults:
        free_energies = extract_free_energies(results)
        if not free_energies:
            raise NoResultsError("no ligand pair has both of its legs")

        connected, discarded = _largest_component(free_energies)
        if discarded:
            logger.warning("estimating without %s, disconnected from the network", ", ".join(discarded))

        femap = _femap(connected)
        cycles = _cycle_closures(femap)
        cycle_error, cycle_error_stderr = _mean_stderr(_cycle_closure_errors(cycles))
        edge_error, edge_error_stderr = _mean_stderr([record["uncertainty"] for record in connected])

        dg = _dg_records(femap)
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
