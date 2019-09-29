"""Wrapper for PyPolyChord.

Author: Will Handley (wh260@cam.ac.uk)
Date: November 2018
"""
import os
import numpy


def run_polychord(loglikelihood, prior, dumper, nDims, nlive, root, ndump,
                  num_repeats, seed=-1):
    """Run PolyChord.

    See https://arxiv.org/abs/1506.00171 for more detail

    Parameters
    ----------
    loglikelihood: :obj:`callable`
        probability function taking a single parameter:

        - theta: numpy.array
                 physical parameters, `shape=(nDims,)`

        returning a log-likelihood (float)

    prior: :obj:`callable`
        tranformation function taking a single parameter

        - cube: numpy.array
                hypercube parameters, `shape=(nDims,)`

        returning physical parameters (`numpy.array`)

    dumper: :obj:`callable`
        access function called every nlive iterations giving a window onto
        current live points. Single parameter, no return:

        - live:
               `numpy.array of` live parameters and loglikelihoods,
               `shape=(nlive,nDims+1)`

    nDims: int
        Dimensionality of sampling space

    nlive: int
        Number of live points

    root: str
        base name for output files

    ndump: int
        How many iterations between dumper function calls

    num_repeats: int
        Length of chain to generate new live points

    seed: int
        Seed for sampler. Optional, no default seed.

    """
    import pypolychord
    from pypolychord.settings import PolyChordSettings

    nDerived = 0
    settings = PolyChordSettings(nDims, nDerived)
    settings.base_dir = os.path.dirname(root)
    settings.file_root = os.path.basename(root)
    settings.nlive = nlive
    settings.num_repeats = num_repeats
    settings.do_clustering = True
    settings.read_resume = False
    settings.compression_factor = numpy.exp(-float(ndump)/nlive)
    settings.precision_criterion = 0.01
    settings.seed = seed

    def polychord_loglikelihood(theta):
        return loglikelihood(theta), []

    def polychord_dumper(live, dead, logweights, logZ, logZerr):
        dumper(live[:, :-2], live[:, -1], dead[:, :-2], dead[:, -1])

    pypolychord.run_polychord(polychord_loglikelihood, nDims, nDerived,
                              settings, prior, polychord_dumper)
