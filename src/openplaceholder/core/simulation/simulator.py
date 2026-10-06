import logging
from abc import ABC, abstractmethod
from typing import Any, Iterator, Self

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
    """To be raised when a network contains no transformations to run."""


class UnsupportedProtocolError(Exception):
    """To be raised when a network carries a protocol the simulator cannot run."""


class DisconnectedNetworkError(Exception):
    """To be raised when a network's ligands do not form one connected component."""


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
    """Each transformation as it was executed, paired with its result."""

    def __init__(self, results: list[list[Any]]):
        self.results = results

    def __iter__(self) -> Iterator[tuple[Transformation, ProtocolDAGResult]]:
        """``(transformation, result)`` pairs, in execution order."""
        for transformation, dag_result in self.results:
            yield transformation, dag_result

    def _to_dict(self) -> dict[Any, Any]:
        return {"results": self.results}

    @classmethod
    def _from_dict(cls, dct: dict[Any, Any]) -> Self:
        return cls(**dct)

    @classmethod
    def _defaults(cls) -> dict[Any, Any]:
        return {}

    def __len__(self) -> int:
        return len(self.results)

    def ok(self) -> bool:
        """Whether every transformation completed without failures."""
        if len(self.results) > 0:
            return all(dag_result.ok() for _, dag_result in self.results)
        return False


class Simulator(Module, ABC):

    @abstractmethod
    def _simulate(self, network: AlchemicalNetwork) -> SimulationResults:
        raise NotImplementedError

    def simulate(self, network: AlchemicalNetwork) -> SimulationResults:
        """Run every transformation in ``network``.

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
