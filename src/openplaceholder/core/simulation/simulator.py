import logging
from abc import ABC, abstractmethod
from typing import Any, Iterable, Iterator, Self

import networkx as nx
from gufe import (
    AlchemicalNetwork,
    ChemicalSystem,
    SmallMoleculeComponent,
    Transformation,
)
from gufe.protocols import ProtocolDAGResult
from gufe.tokenization import GufeTokenizable

from openplaceholder.core.configuration import ConfigBase
from openplaceholder.core.interface import Module

logger = logging.getLogger(__name__)


class SimulatorConfigBase(ConfigBase): ...


class EmptyNetworkError(Exception):
    """To be raised when a network contains no ``Transformation`` instances to run."""


class UnsupportedProtocolError(Exception):
    """To be raised when an ``AlchemicalNetwork`` carries a ``Protocol`` the ``Simulator`` cannot run."""


class DisconnectedNetworkError(Exception):
    """To be raised when the ``AlchemicalNetwork`` instance's ligands do not form a fully connected network."""


def _ligand_names(system: ChemicalSystem) -> set[str]:
    return {c.name for c in system.get_components_of_type(SmallMoleculeComponent)}


def _ligand_components(network: AlchemicalNetwork) -> list[set[str]]:
    """Group the network's ligands by connectivity through its transformations."""
    graph = nx.Graph()
    for transformation in network.edges:
        a, b = _ligand_names(transformation.stateA), _ligand_names(transformation.stateB)
        graph.add_nodes_from(a | b)
        graph.add_edges_from((x, y) for x in a for y in b if x != y)
    return [set(component) for component in nx.connected_components(graph)]


class SimulationResults(GufeTokenizable):  # type: ignore
    """``ProtocolDAGResults`` paired with their source ``Transformation``."""

    def __init__(self, results: Iterable[tuple[Transformation, ProtocolDAGResult]]):
        """Initializer for SimulationResults.

        Parameters
        ----------
        results
            An iterable of ``Transformation`` instances and their
            resulting ``ProtocolDAGResult`` instances.
        """
        self._dag_results = []
        self._transformations = []
        self._size = 0
        self._ok = True

        for transformation, pdr in results:
            self._size += 1
            self._dag_results.append(pdr)
            self._transformations.append(transformation)

            if not pdr.ok():
                self._ok = False

        if self._size == 0:
            self._ok = False

    def __iter__(self) -> Iterator[tuple[Transformation, ProtocolDAGResult]]:
        yield from zip(self._transformations, self._dag_results)

    def _to_dict(self) -> dict[Any, Any]:
        return {"_transformations": self._transformations, "_dag_results": self._dag_results}

    @classmethod
    def _from_dict(cls, dct: dict[Any, Any]) -> Self:
        transformations, dag_results = dct["_transformations"], dct["_dag_results"]
        return cls(list(zip(transformations, dag_results, strict=True)))

    @classmethod
    def _defaults(cls) -> dict[Any, Any]:
        return {}

    def __len__(self) -> int:
        return self._size

    def ok(self) -> bool:
        """Whether every transformation completed without failures."""
        return self._ok


class Simulator(Module, ABC):
    """Simulator abstract class.

    Implementations of a Simulator should process an
    ``AlchemicalNetwork`` and return ``SimulationResults``.
    """

    @abstractmethod
    def _simulate(self, network: AlchemicalNetwork) -> SimulationResults: ...

    def simulate(self, network: AlchemicalNetwork) -> SimulationResults:
        """Run every ``Transformation`` in ``network``.

        Parameters
        ----------
        network
            The ``AlchemicalNetwork`` to process.

        Raises
        ------
        EmptyNetworkError
            When the network has no transformations to run.
        DisconnectedNetworkError
            When the network's ligands span more than one connected component.
        """
        if not network.edges:
            raise EmptyNetworkError("AlchemicalNetwork contains no transformations")

        if len(components := _ligand_components(network)) > 1:
            groups = " | ".join(", ".join(sorted(c)) for c in sorted(components, key=len, reverse=True))
            raise DisconnectedNetworkError(
                f"AlchemicalNetwork ligands form {len(components)} disconnected groups: {groups}"
            )
        logger.info("simulating %d transformations using %s", len(network.edges), self.__class__.__name__)
        return self._simulate(network)
