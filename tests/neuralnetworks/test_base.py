import pytest
from pybambi.neuralnetworks.base import Predictor
import numpy

def test_Predictor():
    ntrain = 500
    ndim = 5
    d2 = numpy.random.rand(ntrain, ndim)
    d1 = numpy.random.rand(ntrain)

    with pytest.raises(ValueError):
        Predictor(d2.transpose(), d1)

    with pytest.raises(ValueError):
        Predictor(d2,d2)

    with pytest.raises(ValueError):
        Predictor(d1,d1)

    p = Predictor(d2,d1)

    with pytest.raises(NotImplementedError):
        p(numpy.random.rand(ndim))

