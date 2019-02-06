"""BAMBI management object.

Author: Pat Scott (p.scott@imperial.ac.uk)
Date: Feb 2019
"""

import numpy as np

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
        self._learner = learner
        self._proxy_trained = False
        self._rolling_failure_fraction = 0.0

    def make_learner(self, params, loglikes):
        if self._learner == 'keras':
            return KerasNetInterpolation(params, loglikes)
        elif self._learner == 'nearestneighbor':
            return NearestNeighborInterpolation(params, loglikes)
        else:
            raise NotImplementedError('learner %s is not implemented.' % learner)


    def dumper(self, live_params, live_loglikes, dead_params, dead_loglikes):
        print("-----------------------------")
        print("Use thumper to do stuff here")
        print("live_params is an array of shape ", live_params.shape)
        print("dead_params is an array of shape ", dead_params.shape)
        print("-----------------------------")
        if not self._proxy_trained: #and reached updint/2:
            train_new_learner(np.concatenate(live_params, dead_params), np.concatenate(live_loglikes, dead_loglikes))


    def loglikelihood(self, params):

        # Short circuit to the full likelihood if the proxy is not yet fully trained
        if not self._proxy_trained: return self._loglikelihood(params)

        # Call the learner and check whether its estimate is to be trusted
        candidate_loglikelihood = self._current_learner(params)
        good_enough = logLInRangeOfTrainingData(candidate_likelihood)

        # If the learner can be trusted, use its estimate, otherwise use the original like and update the failure status
        if good_enough:
            return candidate_loglikelihood
        else:
            self._rolling_failure_fraction = (1.0 + (ntrain - 1.0)*self._rolling_failure_fraction)/ntrain
            if self._rolling_failure_fraction > self._failure_tolerance: self._proxy_trained = False
            return self._loglikelihood(params)


    def train_new_learner(self, params, loglikes):
        self.old_learners.append(self.current_learner)
        self.current_learner = make_learner(params, loglikes)
        if self.current_learner.uncertainty() < self._proxy_tolerance: _proxy_trained = True


    def retrain_old_learner(self, learner):
        pass
