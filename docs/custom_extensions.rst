Using extensions
================

OpenPlaceHolder prioritizes custom extensions through a plugin system. This allows for rapid development of new modules without the need to maintain a copy of the OpenPlaceHolder repository.

Custom modules can be declared in the ``config.toml`` by providing a class path in the implementation field. For instance, a custom ``Validator`` could be added to the list of OpenPlaceHolder ``Validator`` implementations by including a new table in the configuration.  

.. code:: toml

   [[selection.validators]]
   implementation = "openplaceholder.impl.validators:PosebustersValidator"

   [[selection.validators]]
   implementation = "openplaceholder.impl.validators:StereoValidator"

   # Add a new validator table
   [[selection.validators]]
   implementation = "mycustommodules.validators:MyCustomValidator"
   option_1 = "custom config data"

Essentially, OpenPlaceHolder makes no distinction between user-defined modules and its own predefined modules.

Creating a module
=================

Using the ``Validator`` module as an example, one only needs to implement a configuration class and two methods.

.. code:: python

    from dataclasses import dataclass

    from openplaceholder.core.selection.validator import ValidatorConfigBase, Validator
    from openplaceholder.core.structure import Structure

    @dataclass(frozen=True)
    class MyCustomValidatorConfig(ValidatorConfigBase):
        option_1: str = "default_option_1_value"


    class MyCustomValidator(Validator):
        """A custom validation module."""

        # This annotation must be present. If absent,
        # initialization will fail.
        _config: MyCustomValidatorConfig

        # any pre-processing steps, catch configuration
        # errors here. Common to all modules.
        def _setup(self) -> None:
            pass

        # Validator specific
	def _validate_structure(self, structure: Structure) -> bool:
            """Custom procedure for single structure validation."""
	    ...

