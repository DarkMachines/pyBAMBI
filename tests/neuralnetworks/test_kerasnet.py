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

    # Test input has better than 5% accuracy
    for i, l in enumerate(logL):
        print(abs(p(params[i]) - l))
        assert abs(p(params[i]) - l) < 0.5

