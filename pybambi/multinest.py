""" Wrapper for PyMultiNest """
from numpy.ctypeslib import as_array


def run_multinest(loglikelihood, prior, dumper, nDims, nlive, root, eff):
    """ Wrapper function to run MultiNest

    See https://arxiv.org/abs/0809.3437 for more detail

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

    eff: float
        Efficiency of MultiNest
    """
    import pymultinest

    def multinest_prior(cube, ndim, nparams):
        return prior(as_array(cube, shape=(nparams,)))

    def multinest_loglikelihood(cube, ndim, nparams):
        return loglikelihood(as_array(cube, shape=(nparams,)))

    def multinest_dumper(nSamples, nlive, nPar,
                         physLive, posterior, paramConstr,
                         maxLogLike, logZ, logZerr, nullcontext):
        dumper(physLive)

    pymultinest.run(multinest_loglikelihood, multinest_prior, nDims,
                    resume=False, verbose=True, dump_callback=multinest_dumper,
                    n_iter_before_update=nlive//10, n_live_points=nlive,
                    outputfiles_basename=root, sampling_efficiency=eff)
