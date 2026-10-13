import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Self

from gufe import (
    ChemicalSystem,
    ProteinComponent,
    Transformation,
)
from gufe.protocols import ProtocolDAGResult
from gufe.tokenization import GufeTokenizable
from pint import Quantity

from openplaceholder.core.configuration import ConfigBase
from openplaceholder.core.interface import Module
from openplaceholder.core.simulation.simulator import (
    SimulationResults,
    _ligand_names,
    _phase,
)

logger = logging.getLogger(__name__)


class AnalyzerConfigBase(ConfigBase): ...


class NoResultsError(Exception):
    """When ``SimulationResults`` holds nothing that can be analyzed."""


def _leg_estimate(transformation: Transformation, dag_result: ProtocolDAGResult) -> tuple[Quantity, Quantity]:
    """Predicted free energy of one leg, paired with its uncertainty."""
    label = transformation.name or transformation.key
    result = transformation.protocol.gather([dag_result])
    estimate: Quantity = result.get_estimate()
    uncertainty: Quantity = result.get_uncertainty()
    individual = result.get_individual_estimates()

    if uncertainty.m == 0.0:
        # a single repeat has no unc so fall back to MBAR error
        uncertainty = sum(error for _, error in individual) / len(individual)
        logger.debug("%s: no std across repeats, using MBAR error %s", label, uncertainty)
    else:
        logger.debug("%s: using the std across %d repeats, %s", label, len(individual), uncertainty)

    return estimate, uncertainty


def extract_free_energies(results: SimulationResults) -> list[dict[str, Any]]:
    """Pair up the legs of ``results`` into one relative free energy per ligand pair."""
    # ligand pair -> phase -> (free energy, uncertainty).
    legs: dict[tuple[str, str], dict[str, tuple[Quantity, Quantity]]] = defaultdict(dict)

    for transformation, dag_result in results:
        if not dag_result.ok():
            logger.warning("dropping failed leg %s", transformation.name or transformation.key)
            continue

        # the mapper guarantees one ligand per state and one leg per phase
        (ligand_a,) = _ligand_names(transformation.stateA)
        (ligand_b,) = _ligand_names(transformation.stateB)
        legs[(ligand_a, ligand_b)][_phase(transformation.stateA)] = _leg_estimate(transformation, dag_result)

    records = []
    for pair in sorted(legs):
        phases = legs[pair]
        if set(phases) != {"complex", "solvent"}:
            logger.warning("dropping %s -> %s, only the %s leg is usable", *pair, *sorted(phases))
            continue

        complex_dg, complex_error = phases["complex"]
        solvent_dg, solvent_error = phases["solvent"]
        records.append(
            {
                "stateA": pair[0],
                "stateB": pair[1],
                "ddg": complex_dg - solvent_dg,
                "uncertainty": (complex_error**2 + solvent_error**2) ** 0.5,
            }
        )

    return records


class AnalysisResults(GufeTokenizable):  # type: ignore
    """Free energies and network statistics derived from ``SimulationResults``."""

    def __init__(
        self,
        simulation_results: SimulationResults,
        ddg: list[dict[str, Any]],
        dg: list[dict[str, Any]],
        statistics: dict[str, Any],
    ):
        """
        Parameters
        ----------
        simulation_results
            The analyzed ``SimulationResults``.
        ddg
            One record per ligand pair, holding ``stateA``, ``stateB``,
            ``ddg`` and ``uncertainty`` as molar energy quantities.
        dg
            One record per ligand, holding ``ligand``, ``dg`` and
            ``uncertainty`` as molar energy quantities.
        statistics
            Network level diagnostics, e.g. cycle closure.
        """
        self.simulation_results = simulation_results
        self.ddg = ddg
        self.dg = dg
        self.statistics = statistics

    def _to_dict(self) -> dict[Any, Any]:
        return {
            "simulation_results": self.simulation_results,
            "ddg": self.ddg,
            "dg": self.dg,
            "statistics": self.statistics,
        }

    @classmethod
    def _from_dict(cls, dct: dict[Any, Any]) -> Self:
        return cls(**dct)

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
