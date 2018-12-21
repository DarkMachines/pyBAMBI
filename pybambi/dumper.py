"""Function giving access to live points.

Author: Will Handley (wh260@cam.ac.uk)
Date: November 2018
"""


def dumper(live):
    """Dumper function giving access to the live points.

    Parameters
    ----------
    live:
        `numpy.array` of live parameters and loglikelihoods,
        `shape=(nlive,nDims+1)`

    """
    params = live[:, :-1]
    loglikes = live[:, -1]

    print("-----------------------------")
    print("Call neural network code here")
    print(params.shape)
    print(loglikes)
    print("-----------------------------")
