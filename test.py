from pybambi import run_pyBAMBI

from numpy import pi, log, sqrt

nDims = 10

def loglikelihood(theta):
    """ Spherical Gaussian Likelihood """
    sigma = 0.1
    nDims = len(theta)
    logL = -log(2*pi*sigma*sigma)*nDims/2.0 - sum((theta/sigma)**2) /2
    return logL


def prior(cube):
    """ prior mapping [0,1] -> [-1, 1]"""
    return -1 + 2 * cube


run_pyBAMBI(loglikelihood, prior, nDims, nested_sampler='multinest')
run_pyBAMBI(loglikelihood, prior, nDims, nested_sampler='polychord')
