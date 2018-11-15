import os
import pypolychord
from pypolychord.settings import PolyChordSettings

def run_polychord(loglikelihood, prior, dumper, nDims, nlive, root, num_repeats):
    nDerived = 0
    settings = PolyChordSettings(nDims, nDerived)
    settings.base_dir = os.path.dirname(root)
    settings.file_root = os.path.basename(root) 
    settings.nlive = nlive
    settings.num_repeats = num_repeats
    settings.do_clustering = True
    settings.read_resume = False

    def polychord_loglikelihood(theta):
        return loglikelihood(theta), []

    def polychord_dumper(live, dead, logweights, logZ, logZerr):
        dumper(live[:,:-1])

    pypolychord.run_polychord(polychord_loglikelihood, nDims, nDerived, settings, prior, polychord_dumper) 
