"""Diagnostics module for openplaceholder data."""

from io import TextIOBase
from typing import cast

from gufe import AlchemicalNetwork
from gufe.components import SmallMoleculeComponent


def _check_writable(writable: TextIOBase) -> None:
    if not writable.writable():
        raise OSError("Provided IO instance is not writable")


def alchemicalnetwork_to_ligands_sdf(network: AlchemicalNetwork, writable: TextIOBase) -> None:
    """Write the ligands within an ``AlchemicalNetwork`` to the SDF format.

    Parameters
    ----------
    network
        The ``AlchemicalNetwork`` to process.
    writable
        A subclass of TextIOBase, such as a file handle from ``open``.

    Raises
    ------
    OSError
        If the IO instance is not writable.
    """

    _check_writable(writable)

    systems = set()
    for chem_system in network.nodes:
        systems.add(chem_system)

    complex_cs = {cs for cs in network.nodes if "complex" in cs.name}
    for cs in complex_cs:
        smc: SmallMoleculeComponent = cast(SmallMoleculeComponent, cs.components["ligand"])
        writable.write(smc.to_sdf())
