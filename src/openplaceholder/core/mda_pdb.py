"""In-memory MDAnalysis <-> PDB conversion (no disk I/O).

A single home for the ``mda.Writer(StringIO)`` round-trip so callers don't each
re-implement it (and the NUL-byte altLoc fix in particular).
"""

import io

import MDAnalysis as mda


def to_pdb_block(atoms: mda.AtomGroup) -> str:
    """Serialise an AtomGroup to a PDB string in memory."""
    buffer = io.StringIO()
    with mda.Writer(buffer, format="PDB", n_atoms=len(atoms)) as writer:
        writer.write(atoms)
        # the MMCIF parser's default altLoc is a NUL byte that corrupts the
        # fixed-width PDB columns MDAnalysis writes it into and breaks the PDB
        # parsers downstream; scrub it on every round-trip
        return buffer.getvalue().replace("\x00", " ")


def atoms_from_pdb_block(block: str) -> mda.AtomGroup:
    """Parse a PDB string back into an AtomGroup in memory."""
    return mda.Universe(io.StringIO(block), topology_format="PDB").atoms
