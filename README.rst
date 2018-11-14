=======
pyBAMBI
=======

:pyBAMBI: Resurrecting BAMBI for the pythonic deep learning era
:Author: Dark Machines collaboration
:Project organiser: Will Handley
:Version: 0.0.0
:Homepage: https://github.com/williamjameshandley/pyBAMBI
:Collaboration website: https://darkmachines.org

Installation instructions for pypolychord and pymultinest
---------------------------------------------------------

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

(NB MultiNest will be on github later this week).

.. code:: bash
   
   git clone https://github.com/farhanferoz/MultiNest
   cd MultiNest/build
   cmake ..
   make
   cd ../../
   git clone https://github.com/JohannesBuchner/PyMultiNest.git
   cd PyMultiNest
   python setup.py install --user
   cd 
   python -c 'import pymultinest'


Key idea
--------

Use the dumper functions to train a neural network.

To Do
-----

- Establish License
- Unified wrapper for both PolyChord and MultiNest
- Choose neural network strategy
- Continuous integration?
