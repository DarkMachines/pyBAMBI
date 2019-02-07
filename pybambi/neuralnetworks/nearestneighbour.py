"""Nearest neighbour interpolation predictor.

Author: Will Handley (wh260@cam.ac.uk)
Date: November 2018


This implements a nearest neighbour interpolation, and is designed as a
placeholder predictor, rather than an actual neural network
"""
import numpy
from pybambi.neuralnetworks.base import Predictor


class NearestNeighbourInterpolation(Predictor):
    """Nearest Neighbour interpolation.

    Returns the loglikelihood of the training point closest in parameter space

    Parameters
    ----------
    params:
        `numpy.array of` physical parameters to train on
        shape (ntrain, ndims)

    logL:
        `numpy.array` of loglikelihoods to learn
        shape (ntrain,)

    """

    def __init__(self, params, logL):
        """Construct predictor from training data."""
        super(NearestNeighbourInterpolation, self).__init__(params, logL)
        self._params = params[:]
        self._logL = logL[:]

    def __call__(self, x):
        """Calculate proxy loglikelihood.

        Parameters
        ----------
        x:
            `numpy.array` of physical parameters to predict

        Returns
        -------
        proxy loglikelihood value(s)

        """
        distances = numpy.linalg.norm(self._params - x, axis=1)
        i = numpy.argmin(distances)
        return self._logL[i]
