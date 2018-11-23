import pytest
from pybambi.neuralnetworks.nearestneighbor import NearestNeighborInterpolation
import numpy

def test_NearestNeighborInterpolation():
    params = numpy.array([[0,0], [0,1], [1,1], [1,0]])
    logL = numpy.array([1,2,3,4])
    p = NearestNeighborInterpolation(params, logL)
    for t, l in zip(params, logL):
        assert p(t)==l

    assert p([-0.1,-0.1]==1)
    assert p([0.1,-0.1]==1)
    assert p([0.1,0.1]==1)
    assert p([0.1,-0.1]==1)

    assert p([1.1,1.1]==3)
    assert p([0.1,1.1]==2)
    assert p([1.1,0.1]==4)

    

