from unittest import TestCase

from openplaceholder.impl.validators import (
    ClashValidator,
    ClashValidatorConfig,
    PosebustersValidator,
    PosebustersValidatorConfig,
    StereoValidator,
    StereoValidatorConfig,
)


class TestStereoValidator(TestCase):

    def test_init(self) -> None:
        config = StereoValidatorConfig()
        StereoValidator(config)


class TestClashValidator(TestCase):

    def test_init(self) -> None:
        config = ClashValidatorConfig()
        ClashValidator(config)


class TestPosebustersValidator(TestCase):

    def test_init(self) -> None:
        config = PosebustersValidatorConfig()
        PosebustersValidator(config)
