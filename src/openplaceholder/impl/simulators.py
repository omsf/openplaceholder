import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import gufe
from gufe import AlchemicalNetwork
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
    """Runs the ``Transformation`` instances of a  ``AlchemicalNetwork`` sequentially."""

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

    @staticmethod
    def _check_protocol(transformation: gufe.Transformation) -> None:
        """Check that a ``Transformation`` instance's ``Protocol`` is supported.

        Raises
        ------
        UnsupportedProtocolError
        """
        if not isinstance(transformation.protocol, RelativeHybridTopologyProtocol):
            raise UnsupportedProtocolError(
                f"only RelativeHybridTopologyProtocol is supported, "
                f"`{OpenFESimulator._work_name(transformation)}` carries {type(transformation.protocol).__name__}"
            )

    @staticmethod
    def _validate(network: AlchemicalNetwork) -> None:
        """Reject ``AlchemicalNetwork`` instances that will not run correctly.

        Raises
        ------
        UnsupportedProtocolError
            Any ``Transformation`` in the ``AlchemicalNetwork``
            contains an unsupported ``Protocol``.

        ValueError
            The ``Transformation`` names are not unique.
        """

        for transformation in network.edges:
            OpenFESimulator._check_protocol(transformation)

        names = Counter(OpenFESimulator._work_name(t) for t in network.edges)
        if collisions := sorted(name for name, count in names.items() if count > 1):
            # they would share a working directory and overwrite each other
            raise ValueError(f"transformations do not have unique names: {collisions}")

    @staticmethod
    def _work_name(transformation: gufe.Transformation) -> str:
        return transformation.name or str(transformation.key)

    def simulate_transformation(
        self, transformation: gufe.Transformation
    ) -> tuple[gufe.Transformation, ProtocolDAGResult] | None:
        """Run one ``Transformation``, returning itself and its result.

        Parameters
        ----------
        transformation
            The ``Transformation`` to run.

        Returns
        -------
        tuple[``Transformation``, ProtocolDAGResult] | None

        Raises
        ------
        UnsupportedProtocolError
        PermissionError
        """
        self._check_protocol(transformation)
        name = self._work_name(transformation)
        work_directory = Path(self._config.simulation_directory) / name

        logger.info("running transformation %s", name)
        work_directory.mkdir(parents=True, exist_ok=True)
        modified_transformation = transformation.copy_with_replacements(protocol=self._protocol)
        try:
            # carries this simulator's protocol, so its key is the one the
            # result references and it records the settings that actually ran
            dag_result = execute_DAG(
                modified_transformation.create(),
                shared_basedir=work_directory,
                scratch_basedir=work_directory,
                keep_shared=self._config.keep_shared,
                raise_error=False,
            )
        except Exception as e:
            if isinstance(e, TypeError) and "JSON" in repr(e):
                raise e
            logger.exception("skipping transformation %s", name)
            return None

        dag_result.to_json(work_directory / "dag_result.json")

        self._log_summary(name, dag_result)

        return modified_transformation, dag_result

    def _log_summary(self, name: str, dag_result: ProtocolDAGResult) -> None:
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

    def _simulate(self, network: AlchemicalNetwork) -> SimulationResults:
        self._validate(network)
        results = []
        # network.edges is a frozenset; sort for a deterministic execution order
        for transformation in sorted(network.edges, key=lambda t: (t.name or "", str(t.key))):
            ran = self.simulate_transformation(transformation)
            if ran is not None:
                rebuilt, dag_result = ran
                results.append((rebuilt, dag_result))

        return SimulationResults(results)
