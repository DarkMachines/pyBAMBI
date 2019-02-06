from pybambi.neuralnetworks.nearestneighbour import NearestNeighbourInterpolation
import numpy


def test_NearestNeighbourInterpolation():
    params = numpy.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    logL = numpy.array([1, 2, 3, 4])
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
