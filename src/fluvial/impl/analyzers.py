import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, cast

import networkx as nx
import numpy as np
from cinnabar import FEMap, Measurement
from cinnabar.estimators import MLEEstimatorResult
from gufe.protocols import ProtocolDAGResult
from gufe.transformations.transformation import Transformation
from openfe.protocols.openmm_rfe.hybridtop_protocol_results import (
    RelativeHybridTopologyProtocolResult,
)
from openff.units import Quantity, unit

from fluvial.core.analysis.analyzer import (
    AnalysisResults,
    Analyzer,
    AnalyzerConfigBase,
    DDGRecord,
    DGRecord,
    NoResultsError,
)
from fluvial.core.simulation.simulator import (
    SimulationResults,
    _ligand_names,
    _phase,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimulationAnalyzerConfig(AnalyzerConfigBase): ...


def _largest_component(ddg: list[DDGRecord]) -> tuple[list[DDGRecord], set[str]]:
    """Network analysis needs a fully connected network. Keep the largest fully connected network."""
    graph = nx.Graph([(fe[0], fe[1]) for fe in ddg])
    # ties break on the sorted names so the choice does not move between runs
    keep = max(nx.connected_components(graph), key=lambda component: (len(component), sorted(component)))

    connected = []

    for fe in ddg:
        if {fe[0], fe[1]}.issubset(keep):
            connected.append(fe)

    return connected, set(graph.nodes) - keep


class CinnabarAnalyzer(Analyzer):
    """An Analyzer that estimates free energies with cinnabar."""

    _config: SimulationAnalyzerConfig

    def _setup(self) -> None: ...

    def _analyze(self, results: SimulationResults) -> AnalysisResults:
        dropped = set()
        free_energies, _dropped = self.extract_free_energies(results)
        dropped |= _dropped

        if len(free_energies) == 0:
            raise NoResultsError("no ligand pair has both of its legs")

        connected, discarded = _largest_component(free_energies)
        dropped |= discarded

        if len(discarded) > 0:
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

        meta = {}
        meta["statistics"] = {
            "discarded_ligands": list(dropped),
            "cycle_closure_error": cycle_error,
            "cycle_closure_error_stderr": cycle_error_stderr,
            "edge_error": edge_error,
            "edge_error_stderr": edge_error_stderr,
            # rows and columns follow ligand_order, so the two travel together
            "covariance_matrix": estimator.covariance_matrix,
            "ligand_order": estimator.ligand_order,
        }

        return AnalysisResults(results, self.__class__.__name__, free_energies, dg, meta)

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

    @classmethod
    def extract_free_energies(cls, results: SimulationResults) -> tuple[list[DDGRecord], set[str]]:
        """Pair up the legs of ``results`` into one relative free energy per ligand pair."""
        # ligand pair -> phase -> (free energy, uncertainty).
        # e.g. {("A", "B"): {"complex": (dG, std), "solvent": (dG, std)}}
        legs: dict[tuple[str, str], dict[str, tuple[Quantity, Quantity]]] = defaultdict(dict)

        grouped_dags = defaultdict(list)

        for transformation, dag_result in results:
            if not dag_result.ok():
                logger.warning("dropping failed leg %s", transformation.name or transformation.key)
                continue
            grouped_dags[transformation].append(dag_result)

        for transformation, dag_results in grouped_dags.items():
            (ligand_a,) = _ligand_names(transformation.stateA)
            (ligand_b,) = _ligand_names(transformation.stateB)
            legs[(ligand_a, ligand_b)][_phase(transformation.stateA)] = cls._leg_estimate(transformation, dag_results)

        all_ligands = set()
        records: list[tuple[str, str, Quantity, Quantity]] = []
        for lig_a, lig_b in sorted(legs):
            all_ligands.add(lig_a)
            all_ligands.add(lig_b)
            phases = legs[(lig_a, lig_b)]
            if set(phases) != {"complex", "solvent"}:
                logger.warning("dropping %s -> %s, only the %s leg is usable", lig_a, lig_b, *sorted(phases))
                continue

            complex_dg, complex_error = phases["complex"]
            solvent_dg, solvent_error = phases["solvent"]
            ddg: Quantity = cast(Quantity, complex_dg - solvent_dg)  # type: ignore
            uncertainty: Quantity = cast(Quantity, (complex_error**2 + solvent_error**2) ** 0.5)  # type: ignore

            records.append((lig_a, lig_b, ddg, uncertainty))

        remaining_ligands = set()
        for lig_a, lig_b, *_ in records:
            remaining_ligands.add(lig_a)
            remaining_ligands.add(lig_b)
        return records, remaining_ligands ^ all_ligands

    @staticmethod
    def _leg_estimate(
        transformation: Transformation, dag_results: Iterable[ProtocolDAGResult]
    ) -> tuple[Quantity, Quantity]:
        """Predicted free energy of one leg, paired with its uncertainty."""

        result = transformation.protocol.gather(dag_results)
        if not isinstance(result, RelativeHybridTopologyProtocolResult):
            raise TypeError(
                "Expected RelativeHybridTopologyProtocolResult. Protocols other than RelativeHybridTopologyProtocol are not supported."
            )

        label = transformation.name or str(transformation.key)
        estimate: Quantity = result.get_estimate()
        uncertainty: Quantity = result.get_uncertainty()

        if uncertainty.m == 0.0:
            # a single repeat has no uncertainty so fall back to the
            # mean MBAR error. Note that only
            # RelativeHybridTopologyProtocolResult has the
            # `get_individual_estimate` method.
            individuals = result.get_individual_estimates()
            uncertainty = sum(error for _, error in individuals) / len(individuals)
            logger.debug("%s: no std across repeats, using MBAR error %s", label, uncertainty)
        else:
            logger.debug("%s: using the std across %d repeats, %s", label, result.n_protocol_dag_results, uncertainty)

        return estimate, uncertainty
