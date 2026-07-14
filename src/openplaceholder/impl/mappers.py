from dataclasses import dataclass
import logging

from gufe import AlchemicalNetwork, LigandNetwork, Protocol

from openplaceholder.core.assembly.mapper import Mapper, MapperConfigBase
from openplaceholder.core.structure import Structure, LigandPerceptionError

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class LOMAPMapperConfig(MapperConfigBase): ...


class LOMAPMapper(Mapper):

    _config: LOMAPMapperConfig

    def _setup(self) -> None:
        pass

    def _map(self, structures: list[Structure]) -> AlchemicalNetwork:
        raise NotImplementedError

@dataclass(frozen=True)
class KartografMapperConfig(MapperConfigBase):
    pass

class KartografMapper(Mapper):

    _config: KartografMapperConfig

    def _setup(self) -> None:
        pass

    def _map(self, structures: list[Structure]) -> AlchemicalNetwork:
        ligand_network = self._create_ligand_network(structures)
        raise NotImplementedError

    @staticmethod
    def _create_protocol() -> Protocol:
        from openfe import RelativeHybridTopologyProtocol
        return RelativeHybridTopologyProtocol(RelativeHybridTopologyProtocol.default_settings())

    @staticmethod
    def _create_ligand_network(structures: list[Structure]) -> LigandNetwork:
        ligands = []
        logger.debug("building ligand network from %d structures", len(structures))
        for structure in structures:
            try:
                logger.debug("recovering ligand %s from structure", structure.ligand_name)
                mol = structure.to_rdkit_ligand_mol(selection="resname UNL")
            except LigandPerceptionError as e:
                raise e
            logger.debug("added %s to ligand network ligands", structure.ligand_name)
            ligands.append(mol)

        mappers = [openfe.setup.KartografAtomMapper()]
        scorer = openfe.lomap_scorers.default_lomap_score

        network = openfe.ligand_network_planning.generate_minimal_spanning_network(
            ligands = ligands,
            mappers = mappers,
            scorer = scorer,
        )
        return network
