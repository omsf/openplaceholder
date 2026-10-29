import logging
from abc import ABC, abstractmethod

from gufe import AlchemicalNetwork

from openplaceholder.core.configuration import ConfigBase
from openplaceholder.core.interface import Module
from openplaceholder.core.simulation.simulator import _ligand_names, _phase
from openplaceholder.core.structure import StructureSeries

logger = logging.getLogger(__name__)


class MapperConfigBase(ConfigBase): ...


def _validate(network: AlchemicalNetwork) -> None:
    """Reject a network that later stages cannot interpret.

    Raises
    ------
    ValueError
        When a state holds no single ligand, or a ligand pair repeats a phase.
    """
    seen: set[tuple[str, str, str]] = set()

    # network.edges is a frozenset; sort so a rejection always names the same edge
    for transformation in sorted(network.edges, key=lambda t: (t.name or "", str(t.key))):
        label = transformation.name or transformation.key

        pair = []
        for state in (transformation.stateA, transformation.stateB):
            if len(ligands := sorted(_ligand_names(state))) != 1:
                raise ValueError(f"{label}: expected one ligand per state, found {ligands}")
            pair.append(ligands[0])

        leg = (pair[0], pair[1], _phase(transformation.stateA))
        if leg in seen:
            raise ValueError(f"{label}: {leg[0]} -> {leg[1]} already has a {leg[2]} leg")
        seen.add(leg)


class Mapper(Module, ABC):

    @abstractmethod
    def _map(self, structures: StructureSeries) -> AlchemicalNetwork:
        raise NotImplementedError

    def map(self, structures: StructureSeries) -> AlchemicalNetwork:
        """Build an ``AlchemicalNetwork`` from ``structures``.

        Raises
        ------
        ValueError
            When the mapped network has a state that is not of a single ligand.
        """
        logger.info("mapping structures using %s", self.__class__.__name__)
        network = self._map(structures)
        _validate(network)
        return network
