=======
pyBAMBI
=======

:pyBAMBI: Resurrecting BAMBI for the pythonic deep learning era
:Author: Dark Machines collaboration
:Project organiser: Will Handley
:Version: 0.0.0
:Homepage: https://github.com/williamjameshandley/pyBAMBI
:Collaboration website: https://darkmachines.org

Notes
-----

Currently, there is a test script in test.py, which shows how to call pyBAMBI.
The main piece of work to be done is to implement neural network modelling in
``pybambi/dumper.py`` from the view onto the live points every nlive iterations.


Installation instructions
-------------------------

PolyChord Installation
~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash
   
   git clone https://github.com/PolyChord/PolyChordLite
   cd PolyChordLite
   make MPI=
   python setup.py install --user
   cd
   python -c 'import pypolychord'

You may need to adjust the compilers in ``Makefile_gnu``. If in doubt, run ``make veryclean`` to restart. Remove ``MPI=`` to compile with MPI, but be warned, the MPI you link to must be the same as the one that you compiled ``mpi4py`` with.

MultiNest Installation
~~~~~~~~~~~~~~~~~~~~~~

NB:

- MultiNest will be on github later this week.
- PyMultiNest has a two issues:
  1. import conflict with pypolychord ( https://github.com/JohannesBuchner/PyMultiNest/pull/119 )
  2. dumper issue ( https://github.com/JohannesBuchner/PyMultiNest/pull/120 )
- Once these are resolved, one should switch to Johannes Buchner's `PyMultiNest <https://github.com/JohannesBuchner/PyMultiNest.git>`__ 

.. code:: bash
   
   git clone https://github.com/farhanferoz/MultiNest
   cd MultiNest/build
   cmake ..
   make
   cd ../../
   git clone https://github.com/williamjameshandley/PyMultiNest.git
   cd PyMultiNest
   python setup.py install --user
   cd 
   python -c 'import pymultinest'


Key idea
--------

Use the dumper functions to train a neural network.


To Do
-----

- Choose neural network strategy (`Keras <https://keras.io/>`__?)
- Establish License
- Should we make the repository public? (allows `CI <https://docs.python-guide.org/scenarios/ci/>`__ and `PR <https://help.github.com/articles/about-pull-requests/>` workflows for free)
