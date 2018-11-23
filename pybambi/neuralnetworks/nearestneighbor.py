"""Nearest neighbor interpolation predictor.

This implements a nearest neighbor interpolation, and is designed as a
placeholder predictor, rather than an actual neural network
"""
import numpy
from pybambi.neuralnetworks.base import Predictor


class NearestNeighborInterpolation(Predictor):
    """ Nearest Neighbor interpolation
    
    Returns the loglikelihood of the training point closest in parameter space
    """

    def __init__(self, params, logL):
        super(NearestNeighborInterpolation, self).__init__(params, logL)
        self._params = params[:]
        self._logL = logL[:]

    def __call__(self, x):
        distances = numpy.linalg.norm(self._params - x, axis=1)
        i = numpy.argmin(distances)
        return self._logL[i]
