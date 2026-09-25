import logging
from abc import ABC, abstractmethod
from typing import Any, Iterator, Self

import networkx as nx
from gufe import AlchemicalNetwork, ChemicalSystem, SmallMoleculeComponent
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
    """To be raised when a network's ligands do not form one connected component.

    Relative free energies are only meaningful within a connected component: two
    ligands with no path of transformations between them share no reference, so
    their estimates cannot be placed on a common scale.
    """


def _ligand_names(system: ChemicalSystem) -> set[str]:
    return {c.name for c in system.components.values() if isinstance(c, SmallMoleculeComponent)}


def _ligand_components(network: AlchemicalNetwork) -> list[set[str]]:
    """Group the network's ligands by connectivity through its transformations.

    Deliberately not ``AlchemicalNetwork.connected_subgraphs()``: that walks
    ChemicalSystems, and an RBFE network's complex and solvent legs touch
    disjoint sets of them, so a healthy network always reports two subgraphs.
    Connectivity that matters here is between *ligands*.
    """
    graph = nx.Graph()
    for transformation in network.edges:
        a, b = _ligand_names(transformation.stateA), _ligand_names(transformation.stateB)
        graph.add_nodes_from(a | b)
        graph.add_edges_from((x, y) for x in a for y in b if x != y)
    return [set(component) for component in nx.connected_components(graph)]


class SimulationResults(GufeTokenizable):  # type: ignore
    """The executed ProtocolDAGs of an AlchemicalNetwork, with the network as run."""

    def __init__(self, network: AlchemicalNetwork, dag_results: list[ProtocolDAGResult]):
        self.network = network
        self.dag_results = dag_results

    def __iter__(self) -> Iterator[tuple[Any, ProtocolDAGResult]]:
        """``(transformation, result)`` pairs, joined on ``transformation_key``."""
        by_key = {transformation.key: transformation for transformation in self.network.edges}
        for dag_result in self.dag_results:
            transformation = by_key.get(dag_result.transformation_key)
            if transformation is None:
                raise KeyError(f"no transformation in the network matches {dag_result.transformation_key}")
            yield transformation, dag_result

    def _to_dict(self) -> dict[Any, Any]:
        return {"network": self.network, "dag_results": self.dag_results}

    @classmethod
    def _from_dict(cls, dct: dict[Any, Any]) -> Self:
        return cls(**dct)

    @classmethod
    def _defaults(cls) -> dict[Any, Any]:
        return {}

    def __len__(self) -> int:
        return len(self.dag_results)

    def ok(self) -> bool:
        """Whether every transformation completed without failures."""
        return bool(self.dag_results) and all(result.ok() for result in self.dag_results)


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
