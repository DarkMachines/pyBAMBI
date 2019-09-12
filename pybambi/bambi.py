"""Driving routine for pyBAMBI.

Author: Will Handley (wh260@cam.ac.uk)
Date: November 2018
"""
import os
from pybambi.manager import BambiManager


def run_pyBAMBI(loglikelihood, prior, nDims, **kwargs):
    """Run pyBAMBI.

    Parameters
    ----------
    nested_sampler: str
        Choice of nested sampler. Options: `['multinest', 'polychord']`.
        Default `'polychord'`.

    nlive: int
        Number of live points.
        Default `nDims*25`

    root: str
        root of filename.
        Default `'chains/<nested_sampler>'`

    num_repeats: int
        number of repeats for polychord.
        Default `nDims*5`

    eff: float
        efficiency for multinest.
        Default `0.5**nDims`

    learner: object
        information indicating what learning algorithm to use for approximating
        the likelihood. Can be the string `'keras'`, or a `keras.models.Model`
        Default `'keras'`

    ntrain: int
        Number of training points to use
        Default `nlive`

    proxy_tolerance: float
        Required accuracy of proxy.
        Default `0.01`

    ns_output: int
        Nested sampling output level.

    """
    # Process kwargs
    nested_sampler = kwargs.pop('nested_sampler', 'polychord')
    nlive = kwargs.pop('nlive', nDims*25)
    root = kwargs.pop('root', os.path.join('chains', nested_sampler))
    num_repeats = kwargs.pop('num_repeats', nDims*5)
    eff = kwargs.pop('eff', 0.5**nDims)
    learner = kwargs.pop('learner', 'keras')
    proxy_tolerance = kwargs.pop('proxy_tolerance', 0.1)
    failure_tolerance = kwargs.pop('failure_tolerance', 0.5)
    ntrain = kwargs.pop('ntrain', nlive)
    seed = kwargs.pop('seed', -1)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    # Set up the global manager of the BAMBI session.
    thumper = BambiManager(loglikelihood, learner, proxy_tolerance,
                           failure_tolerance, ntrain)

    # Choose and run sampler
    if nested_sampler == 'polychord':
        from pybambi.polychord import run_polychord
        run_polychord(thumper.loglikelihood, prior, thumper.dumper, nDims,
                      nlive, root, ntrain//2, num_repeats, seed)

    elif nested_sampler == 'multinest':
        from pybambi.multinest import run_multinest
        run_multinest(thumper.loglikelihood, prior, thumper.dumper, nDims,
                      nlive, root, ntrain//2, eff, seed)

    else:
        raise NotImplementedError('nested sampler %s is not implemented'
                                  % nested_sampler)
