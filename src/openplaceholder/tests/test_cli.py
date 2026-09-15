from typing import Any

import pytest

from openplaceholder.cli import _enable_smoketest_dropping

_SMOKETEST = "openplaceholder.impl.transformations:ComplexSmoketestTransformation"


def _config_map(*implementations: str) -> dict[str, Any]:
    return {"assembly": {"transformations": [{"implementation": i} for i in implementations]}}


class TestEnableSmoketestDropping:

    def test_sets_drop_failures_on_the_smoketest(self) -> None:
        config_map = _config_map("openplaceholder.impl.transformations:ComplexProtonationTransformation", _SMOKETEST)

        _enable_smoketest_dropping(config_map)

        protonation, smoketest = config_map["assembly"]["transformations"]
        assert smoketest["drop_failures"] is True
        # the flag is about the smoketest alone; nothing else gains a field
        assert "drop_failures" not in protonation

    def test_refuses_a_config_without_a_smoketest(self) -> None:
        # a flag that quietly does nothing is worse than one that says it cannot
        with pytest.raises(SystemExit, match="assembly.transformations"):
            _enable_smoketest_dropping(
                _config_map("openplaceholder.impl.transformations:HeavyAtomAdditionTransformation")
            )

    def test_refuses_a_config_without_any_transformations(self) -> None:
        with pytest.raises(SystemExit):
            _enable_smoketest_dropping({})
