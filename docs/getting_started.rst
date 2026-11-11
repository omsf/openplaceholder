Getting started
===============

``fluvial`` is designed to provide a declaratively composed pipeline for generating the complex structures for a congeneric series, executing physics-based simulations, and returning binding free energy estimates.

As an example, the following configuration file defines a pipeline for a congeneric series of four ligands bound to `TYK2 <https://github.com/OpenFreeEnergy/IndustryBenchmarks2024/tree/main>`_. A full example file can be found `in the source code <https://github.com/omsf/fluvial/blob/main/config.toml>`_.

.. code-block:: toml

   [generation.generator]
   implementation = "fluvial.impl.generator.openfold3:OpenFold3Generator"
   run_openfold_path = "run_openfold"  # must exist in path
   n_diffusion_samples = 5
   generate_n_seeds = 5
   generator_directory = "/tmp/openfold_3_generator/tyk2"
   sequence = "MGSPASDPTVFHKRYLKKIRDLGEGHFGKVSLYCYDPTNDGTGEMVAVKALKADAGPQHRSGWKQEIDILRTLYHEHIIKYKGCCEDAGAASLQLVMEYVPLGSLRDYLPRHSIGLAQLLLFAQQICEGMAYLHAQHYIHRNLAARNVLLDNDRLVKIGDFGLAKAVPEGHEYYRVREDGDSPVFWYAPECLKEYKFYYASDVWSFGVTLYELLTHCDSSQSPPTKFLELIGIAQGQMTVLRLTELLERGERLPRPDKCPAEVYHLMKNCWETEASFRPTFENLIPILKTVHEKYRHHHHHH"

   [generation.generator.ligands]
   ejm_31 = 'CC(=O)Nc1cc(NC(=O)c2c(Cl)cccc2Cl)ccn1'
   ejm_42 = 'CCC(=O)Nc1cc(NC(=O)c2c(Cl)cccc2Cl)ccn1'
   ejm_43 = 'CC(C)C(=O)Nc1cc(NC(=O)c2c(Cl)cccc2Cl)ccn1'
   ejm_44 = 'CC(C)(C)C(=O)Nc1cc(NC(=O)c2c(Cl)cccc2Cl)ccn1'

Structures are **generated** by OpenFold-3 from just the target sequence and the
SMILES of the congeneric series.

.. code-block:: toml

   [[selection.validators]]
   implementation = "fluvial.impl.validators:PosebustersValidator"

   [[selection.validators]]
   implementation = "fluvial.impl.validators:StereoValidator"

These structures are **validated** using `PoseBusters
<https://github.com/maabuu/posebusters>`_ and a simple `RDKit
<https://www.rdkit.org/>`_ stereochemistry validator. Structures that do not
pass validation are simply filtered out of the available pool of structures.

.. code-block:: toml

   [[selection.normalizers]]
   implementation = "fluvial.impl.normalizers:BindingSiteAligner"
   radius = 6.0

The survivors are **normalized** onto a common reference frame by their binding
site.

.. code-block:: toml

   [selection.selector]
   implementation = "fluvial.impl.selector.mpo:MPOSelector"

   [selection.selector.objectives]
   VolumeOverlapObjective = { weight = 1.0 }
   IFPSimilarityObjective = { weight = 1.0 }

A **selector** determines the optimal set of structures that will represent the
congeneric series, in this case optimizing equally on volume and interaction
fingerprint overlaps.

.. code-block:: toml

   [[assembly.transformations]]
   implementation = "fluvial.impl.transformations:MaxVolumeSiteSubstitutionTransformation"

   [[assembly.transformations]]
   implementation = "fluvial.impl.transformations:HeavyAtomAdditionTransformation"

   [[assembly.transformations]]
   implementation = "fluvial.impl.transformations:ComplexProtonationTransformation"
   ph = 7.0

The selected structures are **transformed** such that they all share the same
consistent set of system parameters (e.g. protonation states).

.. code-block:: toml

   [assembly.mapping]
   implementation = "fluvial.impl.mappers:KartografMapper"

A **mapper** sets per-pair atom mappings and plans the alchemical network.

.. code-block:: toml

   [simulation.simulator]
   implementation = "fluvial.impl.simulators:OpenFESimulator"
   simulation_directory = "/tmp/openfe_simulator/simulations"
   production_length_ns = 5.0

The **simulator** runs every transformation in the network.

.. code-block:: toml

   [analysis.analyzer]
   implementation = "fluvial.impl.analyzers:CinnabarAnalyzer"

The **analyzer** takes the simulation outputs and estimates the binding free
energies and reports network level diagnostics.
