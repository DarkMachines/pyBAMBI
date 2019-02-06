"""Function giving access to live points.

Author: Will Handley (wh260@cam.ac.uk)
Date: November 2018
"""


def dumper(live_params, live_loglikes, dead_params, dead_loglikes):
    """Dumper function giving access to the live and dead points.

    Parameters
    ----------
    live_params:
        `numpy.array` of live parameters,
        `shape=(nlive,nDims)`

    live_loglikes:
        `numpy.array` of live loglikelihoods,
        `shape=(nlive)`

    dead_params:
        `numpy.array` of dead parameters,
        `shape=(ndead,nDims)`

    dead_loglikes:
        `numpy.array` of dead loglikelihoods,
        `shape=(nlive)`


    """

    print("-----------------------------")
    print("Call neural network code here")
    print("live_params is an array of shape ", live_params.shape)
    print("dead_params is an array of shape ", dead_params.shape)
    print("-----------------------------")
