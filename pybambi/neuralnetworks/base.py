""" Base predictor class """
import numpy


class Predictor(object):
    """ Base predictor class

    This takes in a training set params -> logL, and aims to construct a
    mapping between them.

    Parameters
    ----------
    params: numpy.array
        Physical parameters to train on
        shape (ntrain, ndims)

    logL:  numpy.array
        loglikelihoods to learn
        shape (ntrain,)
    """
    def __init__(self, params, logL):
        params = numpy.array(params)
        logL = numpy.array(logL)
        if len(params) != len(logL):
            raise ValueError("input and target must be the same length")
        elif params.ndim != 2:
            raise ValueError("input must be two-dimensional")
        elif logL.ndim != 1:
            raise ValueError("target must be one-dimensional")

    def __call__(self, x):
        err = "Predictor: You need to implement a call function"
        raise NotImplementedError(err)
