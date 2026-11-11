import logging
from abc import ABC, abstractmethod
from typing import Any, Self, TypeAlias

from gufe.tokenization import GufeTokenizable
from openff.units.units import Quantity

from fluvial.core.configuration import ConfigBase
from fluvial.core.interface import Module
from fluvial.core.simulation.simulator import (
    SimulationResults,
)

logger = logging.getLogger(__name__)

# ligand_a, ligand_b, ddg, uncertainty
DDGRecord: TypeAlias = tuple[str, str, Quantity, Quantity]

# ligand, dg, uncertainty
DGRecord: TypeAlias = tuple[str, Quantity, Quantity]


class AnalyzerConfigBase(ConfigBase): ...


class NoResultsError(Exception):
    """When ``SimulationResults`` holds nothing that can be analyzed."""


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
