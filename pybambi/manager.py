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

    def __init__(self, learner):
        if (learner == 'keras')
            def self.make_learner():
                return KerasNetInterpolation()
        elif (learner == 'nearestneighbor')
            def self.make_learner():
                return NearestNeighborInterpolation()
        else:
            raise NotImplementedError('Specified learner is not implemented.')


    def get_loglikelihood(loglikelihood, theta):
    # needs to test if prediction is good enough, then call self.learner(theta), otherwise call likelihood(theta)


    def notify_of_new_training_data(live_par, live_lnl, dead_par, dead_lnl):


    def train_new_learner():
        self.old_learners.append(self.current_learner)
        self.current_learner = make_learner()

    def retrain_old_learner(learner):