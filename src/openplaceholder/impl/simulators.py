import logging
from dataclasses import dataclass
from pathlib import Path

from gufe import AlchemicalNetwork
from gufe import Transformation as GufeTransformation
from gufe.protocols.protocoldag import execute_DAG
from gufe.settings import Settings
from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
from openff.units import unit

from openplaceholder.core.simulation.simulator import (
    SimulationResults,
    Simulator,
    SimulatorConfigBase,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenFESimulatorConfig(SimulatorConfigBase):
    simulation_directory: str | Path
    production_length_ns: float = 5.0
    equilibration_length_ns: float = 1.0
    small_molecule_forcefield: str = "openff-2.2.0.offxml"
    protocol_repeats: int = 1
    keep_shared: bool = True


class OpenFESimulator(Simulator):
    """Runs an AlchemicalNetwork's transformations sequentially.

    OpenMM picks the fastest available platform itself, so no device is
    selected here.
    """

    _config: OpenFESimulatorConfig

    def _setup(self) -> None:
        self._protocol = RelativeHybridTopologyProtocol(settings=self._settings())

    def _settings(self) -> Settings:
        settings = RelativeHybridTopologyProtocol.default_settings().unfrozen_copy()
        settings.forcefield_settings.small_molecule_forcefield = self._config.small_molecule_forcefield
        settings.thermo_settings.temperature = 298.15 * unit.kelvin
        settings.thermo_settings.pressure = 1 * unit.bar
        settings.solvation_settings.box_shape = "dodecahedron"
        settings.alchemical_settings.softcore_LJ = "gapsys"
        settings.alchemical_settings.turn_off_core_unique_exceptions = False
        settings.simulation_settings.equilibration_length = self._config.equilibration_length_ns * unit.nanoseconds
        settings.simulation_settings.production_length = self._config.production_length_ns * unit.nanoseconds
        settings.simulation_settings.time_per_iteration = 1 * unit.picoseconds
        settings.protocol_repeats = self._config.protocol_repeats
        return settings

    def _rebuild(self, transformation: GufeTransformation) -> GufeTransformation:
        """Swap in this simulator's protocol, leaving the chemistry untouched."""
        return GufeTransformation(
            stateA=transformation.stateA,
            stateB=transformation.stateB,
            protocol=self._protocol,
            mapping=transformation.mapping,
            name=transformation.name,
        )

    def _simulate(self, network: AlchemicalNetwork) -> SimulationResults:
        root = Path(self._config.simulation_directory)

        dag_results = []
        # network.edges is a frozenset; sort for a deterministic execution order
        for transformation in sorted(network.edges, key=lambda t: (t.name or "", str(t.key))):
            name = transformation.name or str(transformation.key)
            work_directory = root / name
            work_directory.mkdir(parents=True, exist_ok=True)

            logger.info("running transformation %s", name)
            dag_result = execute_DAG(
                self._rebuild(transformation).create(),
                shared_basedir=work_directory,
                scratch_basedir=work_directory,
                keep_shared=self._config.keep_shared,
                # a failed edge should not prevent the rest of the network
                raise_error=False,
            )
            dag_results.append(dag_result)
            try:
                dag_result.to_json(work_directory / "dag_result.json")
            except TypeError as exc:
                # a failure whose exception carries a non-JSON-serialisable
                # argument cannot be written out; losing one edge's file is far
                # better than ending a run that takes days
                logger.warning("could not persist result for %s: %s", name, exc)

            if dag_result.ok():
                estimate = self._protocol.gather([dag_result]).get_estimate()
                logger.info("%s finished: dG = %s", name, estimate)
            else:
                failures = dag_result.protocol_unit_failures
                logger.error("%s failed: %s", name, failures[-1].exception if failures else "unknown")

        return SimulationResults(dag_results)
