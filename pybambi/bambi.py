"""Driving routine for pyBAMBI.

Author: Will Handley (wh260@cam.ac.uk)
Date: November 2018
"""
import os
from pybambi.dumper import dumper


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

    """
    # Process kwargs
    nested_sampler = kwargs.pop('nested_sampler', 'polychord')
    nlive = kwargs.pop('nlive', nDims*25)
    root = kwargs.pop('root', os.path.join('chains', nested_sampler))
    num_repeats = kwargs.pop('num_repeats', nDims*5)
    eff = kwargs.pop('eff', 0.5**nDims)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    # Choose and run sampler
    if nested_sampler == 'polychord':
        from pybambi.polychord import run_polychord
        run_polychord(loglikelihood, prior, dumper, nDims,
                      nlive, root, num_repeats)

    elif nested_sampler == 'multinest':
        from pybambi.multinest import run_multinest
        run_multinest(loglikelihood, prior, dumper, nDims,
                      nlive, root, eff)

    else:
        raise NotImplementedError('nested sampler %s is not implemented'
                                  % nested_sampler)
