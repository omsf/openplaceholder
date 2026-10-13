import pytest
from gufe import AlchemicalNetwork, ChemicalSystem, Transformation

from openplaceholder.core.assembly.mapper import _validate
from openplaceholder.impl.mappers import (
    KartografMapper,
    KartografMapperConfig,
    LOMAPMapper,
    LOMAPMapperConfig,
)
from openplaceholder.tests.helpers import make_leg


def _legs(*pairs: tuple[str, str, str]) -> list[Transformation]:
    return [make_leg(a, b, phase, 0.0)[0] for a, b, phase in pairs]


class TestLOMAPMapper:

    def test_init(self) -> None:
        config = LOMAPMapperConfig()
        LOMAPMapper(config)


class TestKartographMapper:

    def test_init(self) -> None:
        config = KartografMapperConfig()
        KartografMapper(config)


class TestValidate:

    def test_accepts_a_well_formed_network(self) -> None:
        _validate(AlchemicalNetwork(_legs(("a", "b", "complex"), ("a", "b", "solvent"))))

    def test_rejects_a_repeated_phase(self) -> None:
        """Two differently named legs can still be the same pair and phase."""
        (leg,) = _legs(("a", "b", "solvent"))

        with pytest.raises(ValueError, match="already has a solvent leg"):
            _validate(AlchemicalNetwork([leg, leg.copy_with_replacements(name="duplicate")]))

    def test_rejects_a_state_without_a_ligand(self) -> None:
        (leg,) = _legs(("a", "b", "solvent"))
        bare = ChemicalSystem({k: v for k, v in leg.stateA.components.items() if k != "ligand"})

        with pytest.raises(ValueError, match=r"expected one ligand per state, found \[\]"):
            _validate(AlchemicalNetwork([leg.copy_with_replacements(stateA=bare)]))
