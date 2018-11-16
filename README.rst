=======
pyBAMBI
=======

:pyBAMBI: Resurrecting BAMBI for the pythonic deep learning era
:Author: Dark Machines collaboration
:Organiser: Will Handley
:Version: 0.0.0
:GitHub: https://github.com/williamjameshandley/pyBAMBI
:Website: https://darkmachines.org
:Paper: https://arxiv.org/abs/1110.2997

Notes
-----

Currently, there is a test script in test.py, which shows how to call pyBAMBI.
The main piece of work to be done is to implement neural network modelling in
``pybambi/dumper.py`` from the view onto the live points every nlive iterations.


Installation instructions
-------------------------
- `MultiNest installation <https://github.com/williamjameshandley/pyBAMBI/wiki/MultiNest-installation>`__
- `PolyChord installation <https://github.com/williamjameshandley/pyBAMBI/wiki/PolyChord-installation>`__

Key idea
--------

Use the dumper functions to train a neural network.

To Do
-----

- Choose neural network strategy (`Keras <https://keras.io/>`__?)
- Establish License
- Should we make the repository public? (allows `CI <https://docs.python-guide.org/scenarios/ci/>`__ and `PR <https://help.github.com/articles/about-pull-requests/>`__ workflows for free)
