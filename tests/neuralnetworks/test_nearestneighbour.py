import numpy
from pybambi.neuralnetworks.nearestneighbour \
        import NearestNeighbourInterpolation


def test_NearestNeighbourInterpolation():
    numpy.random.seed(0)
    params = numpy.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    params = numpy.repeat(params, 10, axis=0)
    logL = numpy.array([1, 2, 3, 4])
    logL = numpy.repeat(logL, 10)
    p = NearestNeighbourInterpolation(params, logL)
    for t, l in zip(params, logL):
        assert p(t) == l

    assert p([-0.1, -0.1]) == 1
    assert p([0.1, -0.1]) == 1
    assert p([0.1, 0.1]) == 1
    assert p([0.1, -0.1]) == 1

    assert p([1.1, 1.1]) == 3
    assert p([0.1, 1.1]) == 2
    assert p([1.1, 0.1]) == 4
