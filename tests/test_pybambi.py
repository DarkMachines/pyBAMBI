import pytest
import pybambi
import numpy


def test_run_pyBAMBI_inputs():
    with pytest.raises(NotImplementedError):
        pybambi.run_pyBAMBI(0, 0, 0, nested_sampler='NAN')

    with pytest.raises(TypeError):
        pybambi.run_pyBAMBI(0, 0, 0, wrong_kwarg='foo')


def loglikelihood(theta):
    """ Spherical Gaussian log-likelihood.

    Normalised to give a log evidence of 0.
    """
    sigma = 0.1
    nDims = len(theta)
    logL = - numpy.log(2*numpy.pi*sigma*sigma)*nDims/2.0
    logL += -sum((theta/sigma)**2) / 2
    logL += numpy.log(2)*nDims
    loglikelihood.called = True
    return logL


def prior(cube):
    """ prior mapping [0,1] -> [-1, 1]"""
    prior.called = True
    return -1 + 2 * cube


nDims = 3


def test_run_pyBAMBI_multinest():
    numpy.random.seed(0)
    loglikelihood.called = False
    prior.called = False
    pybambi.run_pyBAMBI(loglikelihood, prior, nDims,
                        nested_sampler='multinest',
                        root='chains/multinest', nlive=50,
                        seed=0)
    assert(loglikelihood.called is True)
    assert(prior.called is True)


def test_run_pyBAMBI_polychord():
    numpy.random.seed(0)
    loglikelihood.called = False
    prior.called = False
    pybambi.run_pyBAMBI(loglikelihood, prior, nDims,
                        nested_sampler='polychord',
                        root='chains/polychord', nlive=50,
                        seed=0)
    assert(loglikelihood.called is True)
    assert(prior.called is True)
