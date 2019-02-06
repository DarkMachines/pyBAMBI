"""BAMBI management object.

Author: Pat Scott (p.scott@imperial.ac.uk)
Date: Feb 2019
"""

from pybambi.neuralnetworks.kerasnet import KerasNetInterpolation
from pybambi.neuralnetworks.nearestneighbour import NearestNeighborInterpolation


class BambiManager(object):
    """Does all the talking for BAMBI.

    Takes a new set of training data from the dumper and trains (or retrains) a neural net,
    and assesses whether or not it can be used for a given parameter combination.

    Parameters
    ----------

    """

    old_learners = []

    def __init__(self, loglikelihood, learner):
        self._loglikelihood = loglikelihood
        if (learner == 'keras')
            def self.make_learner():
                return KerasNetInterpolation()
        elif (learner == 'nearestneighbor')
            def self.make_learner():
                return NearestNeighborInterpolation()
        else:
            raise NotImplementedError('Specified learner is not implemented.')


    def dumper(self, live_params, live_loglikes, dead_params, dead_loglikes):
        print("-----------------------------")
        print("Use thumper to do stuff here")
        print("live_params is an array of shape ", live_params.shape)
        print("dead_params is an array of shape ", dead_params.shape)
        print("-----------------------------")


    def get_loglikelihood(loglikelihood, theta):
        # Do some kind of logic to determine if proxy is good enough
        good_enough = False
        if good_enough:
            # Use proxy
            pass
        else:
            return self._loglikelihood(theta)


    def notify_of_new_training_data(live_par, live_lnl, dead_par, dead_lnl):


    def train_new_learner():
        self.old_learners.append(self.current_learner)
        self.current_learner = make_learner()

    def retrain_old_learner(learner):
