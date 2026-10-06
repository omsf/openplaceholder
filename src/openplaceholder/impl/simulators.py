import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from gufe import AlchemicalNetwork
from gufe import Transformation as GufeTransformation
from gufe.protocols import ProtocolDAGResult
from gufe.protocols.protocoldag import execute_DAG
from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
from openfe.protocols.openmm_rfe.equil_rfe_settings import (
    RelativeHybridTopologyProtocolSettings,
)
from openff.units import unit

from openplaceholder.core.simulation.simulator import (
    SimulationResults,
    Simulator,
    SimulatorConfigBase,
    UnsupportedProtocolError,
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

    def _settings(self) -> RelativeHybridTopologyProtocolSettings:
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
        """Swap in this simulator's protocol, leaving everything else untouched."""
        rebuilt: GufeTransformation = transformation.copy_with_replacements(protocol=self._protocol)
        return rebuilt

    def _check_protocol(self, transformation: GufeTransformation) -> None:
        """Checked per transformation, since one can be run on its own."""
        if not isinstance(transformation.protocol, RelativeHybridTopologyProtocol):
            raise UnsupportedProtocolError(
                f"only RelativeHybridTopologyProtocol is supported, "
                f"`{self._work_name(transformation)}` carries {type(transformation.protocol).__name__}"
            )

    def _validate(self, network: AlchemicalNetwork) -> None:
        """Reject networks this simulator would otherwise run incorrectly."""
        for transformation in network.edges:
            self._check_protocol(transformation)

        names = Counter(self._work_name(t) for t in network.edges)
        if collisions := sorted(name for name, count in names.items() if count > 1):
            # they would share a working directory and overwrite each other
            raise ValueError(f"transformations do not have unique names: {collisions}")

    @staticmethod
    def _work_name(transformation: GufeTransformation) -> str:
        return transformation.name or str(transformation.key)

    def simulate_transformation(
        self, transformation: GufeTransformation
    ) -> tuple[GufeTransformation, ProtocolDAGResult] | None:
        """Run one transformation, returning it as executed with its result."""
        self._check_protocol(transformation)
        name = self._work_name(transformation)
        work_directory = Path(self._config.simulation_directory) / name

        logger.info("running transformation %s", name)
        work_directory.mkdir(parents=True, exist_ok=True)
        rebuilt = self._rebuild(transformation)
        try:
            # carries this simulator's protocol, so its key is the one the
            # result references and it records the settings that actually ran
            dag_result = execute_DAG(
                rebuilt.create(),
                shared_basedir=work_directory,
                scratch_basedir=work_directory,
                keep_shared=self._config.keep_shared,
                raise_error=False,
            )
        except Exception:
            logger.exception("skipping transformation %s", name)
            return None

        try:
            dag_result.to_json(work_directory / "dag_result.json")
        except TypeError as exc:
            # a failure whose exception carries a non-JSON-serialisable
            # argument cannot be written out
            logger.warning("could not persist result for %s: %s", name, exc)

        if dag_result.ok():
            try:
                estimate = self._protocol.gather([dag_result]).get_estimate()
                logger.info("%s finished: dG = %s", name, estimate)
            except Exception:
                # reporting only; the result is already stored either way
                logger.exception("%s finished but could not be summarised", name)
        else:
            failures = dag_result.protocol_unit_failures
            logger.error("%s failed: %s", name, failures[-1].exception if failures else "unknown")

        return rebuilt, dag_result

    def _simulate(self, network: AlchemicalNetwork) -> SimulationResults:
        self._validate(network)

        results = []
        # network.edges is a frozenset; sort for a deterministic execution order
        for transformation in sorted(network.edges, key=lambda t: (t.name or "", str(t.key))):
            ran = self.simulate_transformation(transformation)
            if ran is None:
                continue
            rebuilt, dag_result = ran
            results.append([rebuilt, dag_result])

        return SimulationResults(results)
