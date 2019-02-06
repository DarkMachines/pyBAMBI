"""BAMBI management object.

Author: Pat Scott (p.scott@imperial.ac.uk)
Date: Feb 2019
"""


class BambiManager(object):
    """Does all the talking for BAMBI.

    Takes a new set of training data from the dumper and trains (or retrains) a neural net,
    and assesses whether or not it can be used for a given parameter combination.

    Parameters
    ----------

    """

    def __init__(self, loglikelihood, learner):
       self.learner = learner
       self._loglikelihood = loglikelihood

    def dumper(self, live_params, live_loglikes, dead_params, dead_loglikes):
        print("-----------------------------")
        print("Use thumper to do stuff here")
        print("live_params is an array of shape ", live_params.shape)
        print("dead_params is an array of shape ", dead_params.shape)
        print("-----------------------------")

    def loglikelihood(self, theta):
        # Do some kind of logic to determine if proxy is good enough
        good_enough = False
        if good_enough:
            # Use proxy
            pass
        else:
            return self._loglikelihood(theta)
