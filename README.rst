=======
pyBAMBI
=======

:pyBAMBI: Resurrecting BAMBI for the pythonic deep learning era
:Author: Dark Machines collaboration
:Organiser: Will Handley
:Version: 0.0.0
:GitHub: https://github.com/DarkMachines/pyBAMBI
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

Notes
-----

Currently, there is a test script in test.py, which shows how to call pyBAMBI.
The main piece of work to be done is to implement neural network modelling in
``pybambi/dumper.py`` from the view onto the live points every nlive iterations.


Installation instructions
-------------------------
- `MultiNest installation <https://github.com/DarkMachines/pyBAMBI/wiki/MultiNest-installation>`__
- `PolyChord installation <https://github.com/DarkMachines/pyBAMBI/wiki/PolyChord-installation>`__

You can run the tests with:

.. code:: bash
   python -m pytest tests

or 

.. code:: bash
   python setup.py test

Key idea
--------

Use the dumper functions to train a neural network.

To Do
-----

- Choose neural network strategy (`Keras <https://keras.io/>`__?)
- Establish License
- Should we make the repository public? (allows `CI <https://docs.python-guide.org/scenarios/ci/>`__ and `PR <https://help.github.com/articles/about-pull-requests/>`__ workflows for free)
