import logging
from dataclasses import dataclass
from typing import Any

import networkx as nx
from cinnabar import FEMap, Measurement
from openff.units import unit

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
    """network analysis needs a fully connected network: keep the largest fc network."""
    graph = nx.Graph([(record["stateA"], record["stateB"]) for record in ddg])
    # ties break on the sorted names so the choice does not move between runs
    keep = max(nx.connected_components(graph), key=lambda component: (len(component), sorted(component)))
    connected = [record for record in ddg if record["stateA"] in keep and record["stateB"] in keep]
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
        statistics = {
            "n_ligands": femap.n_ligands,
            "n_edges": femap.n_edges,
            "discarded_ligands": discarded,
        }

        return AnalysisResults(results, free_energies, _dg_records(femap), statistics)
