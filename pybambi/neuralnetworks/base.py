"""Base predictor class.

Author: Will Handley (wh260@cam.ac.uk)
Date: November 2018
"""
import numpy


class Predictor(object):
    """Base predictor class.

    This takes in a training set params -> logL, and aims to construct a
    mapping between them.

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
        params = numpy.array(params)
        logL = numpy.array(logL)
        if len(params) != len(logL):
            raise ValueError("input and target must be the same length")
        elif params.ndim != 2:
            raise ValueError("input must be two-dimensional")
        elif logL.ndim != 1:
            raise ValueError("target must be one-dimensional")

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
        err = "Predictor: You need to implement a call function"
        raise NotImplementedError(err)
