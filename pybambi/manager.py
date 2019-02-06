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

    def __init__(self, learner):
       self.learner = learner()



    def get_loglikelihood(loglikelihood, theta):
    # needs to test if prediction is good enough, then call self.learner(theta), otherwise call likelihood(theta)