import unittest

from openplaceholder.impl.validators import StereoValidator, StereoValidatorConfig


class TestStereoValidator(unittest.TestCase):

    def test_config_defaults(self) -> None:
        config = StereoValidatorConfig()
        assert config.require_inchi_match is True
        assert config.require_smiles_match is True
