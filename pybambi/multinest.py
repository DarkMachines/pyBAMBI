import pymultinest
import numpy
from numpy.ctypeslib import as_array

def run_multinest(loglikelihood, prior, dumper, nDims):

    def multinest_prior(cube, ndim, nparams):
        return prior(as_array(cube,shape=(nparams,)))

    def multinest_loglikelihood(cube, ndim, nparams):
        return loglikelihood(as_array(cube,shape=(nparams,)))

    def multinest_dumper(nSamples,nlive,nPar,
                         physLive,posterior,paramConstr,
                         maxLogLike,logZ,logZerr,nullcontext):
        dumper(physLive)


    pymultinest.run(multinest_loglikelihood, multinest_prior, nDims, resume=False, verbose=True, dump_callback=multinest_dumper)
