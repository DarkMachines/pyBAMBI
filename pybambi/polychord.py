""" Wrapper for PyPolyChord """

import os
import pypolychord
from pypolychord.settings import PolyChordSettings


def run_polychord(loglikelihood, prior, dumper, nDims, nlive, root,
                  num_repeats):
    """ Wrapper function to run PolyChord

    See https://arxiv.org/abs/1506.00171 for more detail

    Parameters
    ----------
    loglikelihood: callable
        probability function taking a single parameter:

        - theta: numpy.array
                 physical parameters, `shape=(nDims,)`

        returning a log-likelihood (float)

    prior: callable
        tranformation function taking a single parameter

        - cube: numpy.array 
                hypercube parameters, `shape=(nDims,)`

        returning physical parameters (`numpy.array`) 

    dumper: callable
        access function called every nlive iterations giving a window onto
        current live points. Single parameter, no return:

        - live: numpy.array
               live parameters and loglikelihoods, `shape=(nlive,nDims+1)` 
                  
    nDims: int
        Dimensionality of sampling space

    nlive: int
        Number of live points

    root: str
        base name for output files

    repeats: int
        Length of chain to generate new live points
    """
    nDerived = 0
    settings = PolyChordSettings(nDims, nDerived)
    settings.base_dir = os.path.dirname(root)
    settings.file_root = os.path.basename(root)
    settings.nlive = nlive
    settings.num_repeats = repeats
    settings.do_clustering = True
    settings.read_resume = False

    def polychord_loglikelihood(theta):
        return loglikelihood(theta), []

    def polychord_dumper(live, dead, logweights, logZ, logZerr):
        dumper(live[:, :-1])

    pypolychord.run_polychord(polychord_loglikelihood, nDims, nDerived,
                              settings, prior, polychord_dumper)
