import pytest
import pybambi
import numpy
import shutil

def test_run_pyBAMBI_inputs():
    with pytest.raises(NotImplementedError):
        pybambi.run_pyBAMBI(0, 0, 0, nested_sampler='NAN')

    with pytest.raises(TypeError):
        pybambi.run_pyBAMBI(0, 0, 0, wrong_kwarg='foo')


nDims = 3

def loglikelihood(theta):
    """ Spherical Gaussian log-likelihood.
    
    Normalised to give a log evidence of 0.
    """
    sigma = 0.1
    nDims = len(theta)
    logL = -numpy.log(2*numpy.pi*sigma*sigma)*nDims/2.0 - sum((theta/sigma)**2) /2 + numpy.log(2)*nDims
    return logL


def prior(cube):
    """ prior mapping [0,1] -> [-1, 1]"""
    return -1 + 2 * cube


def test_run_pyBAMBI_multinest():
    pybambi.run_pyBAMBI(loglikelihood, prior, nDims, nested_sampler='multinest', root='.chains/polychord', nlive=50)
    shutil.rmtree('.chains')


def test_run_pyBAMBI_polychord():
    pybambi.run_pyBAMBI(loglikelihood, prior, nDims, nested_sampler='polychord', root='.chains/multinest', nlive=50)
    shutil.rmtree('.chains')
