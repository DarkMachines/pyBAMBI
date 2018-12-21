import numpy
import scipy.stats
from pybambi.neuralnetworks.kerasnet import KerasNetInterpolation


def test_KerasNet():
    numpy.random.seed(0)
    # Generate data from a 2D Gaussian
    # Pick some random numbers from a multivariate Gaussian
    # (gives reasonable sampling density in the interesting regions)
    mu = numpy.array([0, 0])
    sigma = 1
    Sigma = numpy.array([[sigma, 0], [0, sigma]])
    dist = scipy.stats.multivariate_normal(mu, Sigma)

    params = dist.rvs(1000)
    logL = dist.logpdf(params)

    # Initialise the keras net example, which includes training
    p = KerasNetInterpolation(params, logL)

    # Test 1D input has better than 5% accuracy
    assert numpy.abs((p(params[0]) - logL[0]) / logL[0]) < 0.05

    # Test 2D input has better than 5% accuracy
    assert numpy.abs((p(params) - logL) / logL).max() < 0.05
