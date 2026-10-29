import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Self, TypeAlias, cast

from gufe import Transformation
from gufe.protocols import ProtocolDAGResult
from gufe.tokenization import GufeTokenizable
from openff.units.units import Quantity

from openplaceholder.core.configuration import ConfigBase
from openplaceholder.core.interface import Module
from openplaceholder.core.simulation.simulator import (
    SimulationResults,
    _ligand_names,
    _phase,
)

logger = logging.getLogger(__name__)

# ligand_a, ligand_b, ddg, uncertainty
DDGRecord: TypeAlias = tuple[str, str, Quantity, Quantity]

# ligand, dg, uncertainty
DGRecord: TypeAlias = tuple[str, Quantity, Quantity]


class AnalyzerConfigBase(ConfigBase): ...


class NoResultsError(Exception):
    """When ``SimulationResults`` holds nothing that can be analyzed."""


def _leg_estimate(transformation: Transformation, dag_result: ProtocolDAGResult) -> tuple[Quantity, Quantity]:
    """Predicted free energy of one leg, paired with its uncertainty."""
    label = transformation.name or str(transformation.key)
    result = transformation.protocol.gather([dag_result])
    estimate: Quantity = result.get_estimate()
    uncertainty: Quantity = result.get_uncertainty()

    if uncertainty.m == 0.0:
        # a single repeat has no uncertainty so fall back to MBAR
        # error. Note that only RelativeHybridTopologyProtocolResult
        # has the `get_individual_estimate` method.
        individual = result.get_individual_estimates()
        uncertainty = sum(error for _, error in individual) / len(individual)
        logger.debug("%s: no std across repeats, using MBAR error %s", label, uncertainty)
    else:
        logger.debug("%s: using the std across %d repeats, %s", label, result.n_protocol_dag_results, uncertainty)

    return estimate, uncertainty


def extract_free_energies(results: SimulationResults) -> tuple[list[DDGRecord], set[str]]:
    """Pair up the legs of ``results`` into one relative free energy per ligand pair."""
    # ligand pair -> phase -> (free energy, uncertainty).
    # e.g. {("A", "B"): {"complex": (dG, std), "solvent": (dG, std)}}
    legs: dict[tuple[str, str], dict[str, tuple[Quantity, Quantity]]] = defaultdict(dict)

    for transformation, dag_result in results:
        if not dag_result.ok():
            logger.warning("dropping failed leg %s", transformation.name or transformation.key)
            continue

        # the mapper guarantees one ligand per state and one leg per phase
        (ligand_a,) = _ligand_names(transformation.stateA)
        (ligand_b,) = _ligand_names(transformation.stateB)
        legs[(ligand_a, ligand_b)][_phase(transformation.stateA)] = _leg_estimate(transformation, dag_result)

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


class AnalysisResults(GufeTokenizable):  # type: ignore
    """Free energies and network statistics derived from ``SimulationResults``."""

    def __init__(
        self,
        simulation_results: SimulationResults,
        analyzer_name: str,
        ddg: list[DDGRecord],
        dg: list[DGRecord],
        meta: dict[str, Any],
    ):
        """
        Parameters
        ----------
        simulation_results
            The analyzed ``SimulationResults``.

        analyzer_name
            Name of the analyzer which produced the results.

        ddg
            One record per ligand pair, holding ``stateA``, ``stateB``,
            ``ddg`` and ``uncertainty`` as molar energy quantities.
        dg
            One record per ligand, holding ``ligand``, ``dg`` and
            ``uncertainty`` as molar energy quantities.
        meta
            Network level diagnostics, e.g. cycle closure.
        """
        self.simulation_results = simulation_results
        self.analyzer_name = analyzer_name
        self._ddg = ddg
        self._dg = dg
        self.meta = meta

    def ddg_pairs(self) -> list[tuple[str, str]]:
        return [record[0:2] for record in self._ddg]

    def ddg(self, lig_a: str, lig_b: str) -> tuple[Quantity, Quantity] | None:
        for record in self._ddg:
            rec_a, rec_b, dif, uncertainty = record
            if (lig_a, lig_b) == (rec_a, rec_b):
                return (dif, uncertainty)
            elif (lig_b, lig_a) == (rec_a, rec_b):
                reversed = -1 * dif.m * uncertainty.units
                return (reversed, uncertainty)
        return None

    def ligands(self) -> set[str]:
        return {lig for lig, _, _ in self._dg}

    def dg(self, ligand: str) -> tuple[Quantity, Quantity] | None:
        for record in self._dg:
            rec_lig, dif, uncertainty = record
            if ligand == rec_lig:
                return (dif, uncertainty)
        return None

    def _to_dict(self) -> dict[Any, Any]:
        return {
            "simulation_results": self.simulation_results,
            "analyzer_name": self.analyzer_name,
            "ddg": self._ddg,
            "dg": self._dg,
            "meta": self.meta,
        }

    @classmethod
    def _from_dict(cls, dct: dict[Any, Any]) -> Self:
        cpy = dct.copy()
        cpy["ddg"] = [(a, b, ddg, unc) for a, b, ddg, unc in cpy["ddg"]]
        cpy["dg"] = [(lig, ddg, unc) for lig, ddg, unc in cpy["dg"]]

        return cls(**cpy)

    @classmethod
    def _defaults(cls) -> dict[Any, Any]:
        return {}


class Analyzer(Module, ABC):
    """Analyzer abstract class.

    Implementations of an Analyzer should process ``SimulationResults``
    and return ``AnalysisResults``.
    """

    @abstractmethod
    def _analyze(self, results: SimulationResults) -> AnalysisResults: ...

    def analyze(self, results: SimulationResults) -> AnalysisResults:
        """Derive free energies and network statistics from ``results``.

        Parameters
        ----------
        results
            The ``SimulationResults`` to analyze.

        Raises
        ------
        NoResultsError
            When ``results`` holds no transformations.
        """
        if len(results) == 0:
            raise NoResultsError("SimulationResults contains no transformations")

        logger.info("analyzing %d transformations using %s", len(results), self.__class__.__name__)
        return self._analyze(results)
