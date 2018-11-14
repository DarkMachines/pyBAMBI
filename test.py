from pybambi import run_pyBAMBI

from numpy import pi, log, sqrt

nDims = 3

def loglikelihood(theta):
    """ Simple Gaussian Likelihood """
    sigma = 0.1
    nDims = len(theta)

    r2 = sum(theta**2)

    logL = -log(2*pi*sigma*sigma)*nDims/2.0
    logL += -r2/2/sigma/sigma

    return logL

def prior(cube):
    """ prior mapping [0,1] -> [-1, 1]"""
    return -1 + 2 * cube


run_pyBAMBI(loglikelihood, prior, nDims, nested_sampler='multinest')
run_pyBAMBI(loglikelihood, prior, nDims, nested_sampler='polychord')
