=======
pyBAMBI
=======

:pyBAMBI: Resurrecting BAMBI for the pythonic deep learning era
:Author: Dark Machines collaboration
:Organiser: Will Handley
:Version: 0.0.0
:GitHub: https://github.com/DarkMachines/pyBAMBI
:Documentation: https://pybambi.readthedocs.io/
:Website: https://darkmachines.org
:Paper: https://arxiv.org/abs/1110.2997

.. image:: https://travis-ci.org/DarkMachines/pyBAMBI.svg?branch=master
   :target: https://travis-ci.org/DarkMachines/pyBAMBI
   :alt: Build Status
.. image:: https://codecov.io/gh/DarkMachines/pyBAMBI/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/DarkMachines/pyBAMBI
   :alt: Test Coverage Status
.. image:: https://readthedocs.org/projects/pybambi/badge/?version=latest
   :target: https://pybambi.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

Quick links
-----------

- `Contributing <https://github.com/DarkMachines/pyBAMBI/blob/master/.github/CONTRIBUTING.md>`__
- `Code of conduct <https://github.com/DarkMachines/pyBAMBI/blob/master/.github/CODE_OF_CONDUCT.md>`__

Notes
-----

Currently, there is a test script in test.py, which shows how to call pyBAMBI.
The main piece of work to be done is to implement neural network modelling in
``pybambi/dumper.py`` from the view onto the live points every nlive iterations.


Installation instructions
-------------------------
- `MultiNest installation <https://github.com/DarkMachines/pyBAMBI/wiki/MultiNest-installation>`__
- `PolyChord installation <https://github.com/DarkMachines/pyBAMBI/wiki/PolyChord-installation>`__

You can clone the repository with

.. code:: bash

   git clone https://github.com/DarkMachines/pyBAMBI 

Although if you wish to
`contribute <https://github.com/DarkMachines/pyBAMBI/blob/master/.github/CONTRIBUTING.md>`__,
you should fork the repository to your own account.

You can run the tests with either of the commands:

.. code:: bash

   python -m pytest tests
   python setup.py test


Key idea
--------

Use the dumper functions to train a neural network.
