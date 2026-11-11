Core Concepts
==============

Pipeline module overview
---------------------------

1. ``StructureGenerator``

   Expects no data (outside of that from ``config.toml``), produces a ``StructureSet``.

   This generates the initial set of structures for a congeneric series. This should include repeats for each protein-ligand pair.

2. ``Validator``

   Chainable, expects a ``StructureSet``, produces a ``StructureSet``.

   This module filters invalid structures based on each structure in isolation. These are, ideally, lightweight checks on the physical validity of each structure. They return a ``StructureSet`` of equal or lesser size compared to the ``StructureSet`` they process.

3. ``Normalizer``

   Chainable, expects a ``StructureSet``, produces a ``StructureSet``.

   These modules are preprocessing operations that perform non-chemical alterations of the structures passed to them. These types of operations may include structure alignment and other processes that make final structure selection easier.

4. ``Selector``

   Expects a ``StructureSet``, produces a ``StructureSeries``

   This module is responsible for determining the final set of structures that will be used for planning and executing a free energy network. For a ``StructureSet`` of ``N`` protein-ligand pairs, the selection should return exactly ``N`` unique structures.

5. ``Transformation``

   Chainable, expects a ``StructureSeries``, produces a ``StructureSeries``.

   These modules apply chemical alterations to the structures it receives, such as assignment of protonation state. These operations are sensitive to the order in which they run. After these steps, there are no more opportunities to modify the structures that will be simulated.

6. ``Mapper``

   Expects a ``StructureSeries``, produces an ``AlchemicalNetwork``.

   The final set of structures for the congeneric series are used to produce an AlchemicalNetwork.

7. ``Simulator``

   Expects an ``AlchemicalNetwork``, produces a ``SimulationResults`` instance.

   From the provided ``AlchemicalNetwork``, perform simulations defined by the network's transformation ``Protocol`` instances. The results are packaged as ``SimulationResults``, which pair the Transformations along with the resulting ``ProtocolDAGResult``.

8. ``Analyzer``

   Expects a ``SimulationResults`` instance, produces an ``AnalysisResults`` instance.

   Processes the ProtocolDAGResults obtained from the simulations and compute ddGs.

Core Versus Implementation
-----------------------------

``fluvial`` defines a set of abstract classes for key modules in the pipeline. These abstract classes are part of the ``fluvial.core`` subpackage. They define the expectations and inter-module data types that tie the pipeline together. This core defines the high level view of how a pipeline is constructed.

Concrete implementations of the core classes are required for a pipeline to be functional. For instance, the ``PosebustersValidator`` class is a concrete implementation of the ``Validator`` abstract class and adheres to the requirements imposed by the ``Validator`` abstract class. The concrete classes and related functionality are part of the ``fluvial.impl`` subpackage. These are just the shipped implementations, but ``fluvial`` allows users to define their own implementations outside of the default implementation subpackage.

Structures and Structure Containers
-----------------------------------

A ``Structure`` is a data container that holds the raw bytes of a structure (for instance PDB), as they would be found on-disk, along with additional metadata such as ligand identity and sequence information. Its purpose is to provide a common interface for interacting with structure data between modules.

Many of the core modules are responsible for manipulating collections of ``Structure`` instances. These collections have different structures and meaning:

1. ``StructureReplicates`` is a ``list`` of ``Structure`` instances that represent the same protein ligand pair. For instance, the different instances may come from repeated inference of a cofolding model for the same complex using different seeds. This allows ensemble-like input for each ligand before validation and selection.
2. ``StructureSet`` is a ``list`` of ``StructureReplicates``. For a congeneric series of ``N`` ligands, we expect ``N`` ``StructureReplicates`` instances. This is the output format for all ``StructureGenerator`` modules and the input format for ``Validator``, ``Normalizer``, and ``Selector`` modules.
3. ``StructureSeries`` is a ``list`` of ``Structure`` instances that represent the working set of of structures representing the congeneric series. It is effectively ``StructureSet`` that has flattened its ``StructureReplicates`` to a single ``Structure``
