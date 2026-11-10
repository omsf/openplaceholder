Installation
============

OpenPlaceHolder is currently only installable via source code and environments are managed through `pixi <https://pixi.prefix.dev/latest/>`_.

To obtain a copy of the source code, use ``git clone`` by running

.. code:: bash

   git clone https://github.com/omsf/openplaceholder


or if you have an SSH key set up with GitHub, run
   
.. code:: bash

   git clone git@github.com:omsf/openplaceholder

The package can then be installed and tested with pixi.

.. code:: bash

   cd openplaceholder
   pixi run pytest src/

This will test that the core functionality of OpenPlaceHolder is operating correctly. In order to generate structures with `openfold3 <https://openfold-3.readthedocs.io/en/latest/>`_, a pre-existing version of openfold must be installed on the system according `their documentation <https://openfold-3.readthedocs.io/en/latest/Installation.html>`_.
