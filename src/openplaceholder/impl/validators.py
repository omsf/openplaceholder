from dataclasses import dataclass

from openplaceholder.core.selection.validator import Validator
from openplaceholder.core.structure.structure import Structure


@dataclass(frozen=True, eq=True)
class PosebustersValidatorConfig:
    pass


class PosebustersValidator(Validator):

    def __init__(self, config: PosebustersValidatorConfig):
        self._config = config

    def validate(self, structures: list[Structure]) -> list[Structure]:
        raise NotImplementedError


@dataclass(frozen=True, eq=True)
class StereoValidatorConfig:
    pass


class StereoValidator(Validator):

    def __init__(self, config: StereoValidatorConfig):
        self._config = config

    def validate(self, structures: list[Structure]) -> list[Structure]:

        passed = []
        failed = []
        for structure in structures:

            original_smiles = structure.smiles
            derived_smiles = self._determine_smiles_from_bytes(structure.structure, structure.structure_format)

            if self._compatible(original_smiles, derived_smiles):
                passed.append(structure)
            else:
                failed.append(structure)

        # NOTE: I plan to extend this to also include failures in the output
        return passed

    def _determine_smiles_from_bytes(self, structure_bytes, structure_format):
        # use whatever reader to take the decoded file contents according to the format specified
        raise NotImplementedError
