"""Protonation interfaces.

Protein and ligand protonation are deliberately separate abstractions: a
protonation method may cover one, the other, or both, and they are applied
independently (see the note in ``ComplexProtonationTransformation`` about the
resulting protein/ligand H-bond mismatch risk). Add a new method by
implementing one or both of these ABCs.
"""

from abc import ABC, abstractmethod

from rdkit import Chem


class LigandProtonator(ABC):
    """Add explicit hydrogens to a ligand ``Chem.Mol`` at a target pH."""

    @abstractmethod
    def protonate(self, mol: Chem.Mol, ph: float) -> Chem.Mol:
        raise NotImplementedError


class ProteinProtonator(ABC):
    """Add hydrogens to a protein, given and returning PDB text, at a target pH."""

    @abstractmethod
    def protonate(self, pdb: str, ph: float) -> str:
        raise NotImplementedError
