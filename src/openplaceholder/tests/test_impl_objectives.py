import base64

import pytest

from openplaceholder.core.structure import Structure
from openplaceholder.impl.selector.objectives.ifp import (
    IFPSimilarityObjective,
    IFPSimilarityObjectiveConfig,
)
from openplaceholder.impl.selector.objectives.volume import (
    VolumeOverlapObjective,
    VolumeOverlapObjectiveConfig,
)
from openplaceholder.tests.datafiles import TYK2_LIG_PDB
from openplaceholder.tests.helpers import read_gzip_file


def _tyk2_structure() -> Structure:
    content = base64.b64encode(read_gzip_file(str(TYK2_LIG_PDB))).decode()
    return Structure(
        sequence=(
            "TVFHKRYLKKIRDLGEGHFGKVSLYCYDPTNDGTGEMVAVKALKADCGPQHRSGWKQEIDILRTLYHEHIIKYKGCCEDQGEKSLQLVMEYVPLGSLR"
            "DYLPRHSIGLAQLLLFAQQICEGMAYLHAQHYIHRDLAARNVLLDNDRLVKIGDFGLAKAVPEGHEYYRVREDGDSPVFWYAPECLKEYKFYYASDVW"
            "SFGVTLYELLTHCDSSQSPPTKFLELIGIAQGQMTVLRLTELLERGERLPRPDKCPCEVYHLMKNCWETEASFRPTFENLIPILKTVHEKYQ"
        ),
        ligand_smiles="[H]c1nc(N([H])C(=O)O[C@@]([H])([H])[H])c([H])c(N([H])C(=O)c2c(Cl)c([H])c([H])c([H])c2Cl)c1[H]",
        # the actual resname embedded in this PDB is the generic "LIG", not a query-derived name
        ligand_name="LIG",
        structure_format="pdb",
        structure_data=content,
    )


class TestVolumeOverlapObjective:

    def test_self_overlap_is_one(self) -> None:
        structure = _tyk2_structure()
        objective = VolumeOverlapObjective(VolumeOverlapObjectiveConfig())
        pytest.approx(objective.score(structure, structure), 1.0)


class TestIFPSimilarityObjective:

    def test_self_similarity_is_one(self) -> None:
        structure = _tyk2_structure()
        objective = IFPSimilarityObjective(IFPSimilarityObjectiveConfig())
        pytest.approx(objective.score(structure, structure), 1.0)
