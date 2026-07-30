import logging
from dataclasses import dataclass
from io import StringIO

import MDAnalysis as mda
import openfe
from gufe import AlchemicalNetwork, LigandNetwork
from openfe.setup.alchemical_network_planner import RBFEAlchemicalNetworkPlanner
from openff.units import unit

from openplaceholder.core.assembly.mapper import Mapper, MapperConfigBase
from openplaceholder.core.structure import Structure

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
        ligand_network: LigandNetwork = self._create_ligand_network(structures)
        solvent = openfe.SolventComponent(
            ion_concentration=0.15 * unit.molar,
            positive_ion="Na",
            negative_ion="Cl",
            neutralize=True,
        )
        protein: openfe.ProteinComponent = self._extract_protein(structures[0])
        # This chooses RelativeHybridTopologyProtocol, along with its
        # default settings, by default
        planner = RBFEAlchemicalNetworkPlanner()
        alchemical_network = planner(
            ligands=ligand_network.nodes,
            solvent=solvent,
            protein=protein,
        )
        return alchemical_network

    @staticmethod
    def _extract_protein(structure: Structure) -> openfe.ProteinComponent:
        """Construct a ProteinComponent from the Structure's embedded structure data."""
        protein_atoms = structure.protein_atoms()
        buffer = StringIO()
        with mda.Writer(buffer, format="PDB", n_atoms=len(protein_atoms)) as writer:
            writer.write(protein_atoms)
            buffer.seek(0)
            contents = openfe.ProteinComponent.from_pdb_file(buffer)
        return contents

    @staticmethod
    def _create_ligand_network(structures: list[Structure]) -> LigandNetwork:
        """Create a ligand network from a list of provided structures.

        Ligands contained in the structures are extracted, converted
        to SmallMoleculeComponents, and used to plan a LigandNetwork
        using the Kartograf atom mapper.
        """
        ligands = []
        logger.debug("building ligand network from %d structures", len(structures))
        for structure in structures:
            logger.debug("recovering ligand %s from structure", structure.ligand_name)
            mol = structure.to_rdkit_ligand_mol()
            smc = openfe.SmallMoleculeComponent(mol, name=structure.ligand_name)
            ligands.append(smc)
            logger.debug("added %s to ligand network ligands", structure.ligand_name)

        mappers = [openfe.setup.KartografAtomMapper()]
        scorer = openfe.lomap_scorers.default_lomap_score

        # TODO: try generate_lomap_network
        network = openfe.ligand_network_planning.generate_minimal_spanning_network(
            ligands=ligands,
            mappers=mappers,
            scorer=scorer,
        )
        return network
