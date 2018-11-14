import pypolychord
from pypolychord.settings import PolyChordSettings

def run_polychord(loglikelihood, prior, dumper, nDims, nlive, root):
    nDerived = 0
    settings = PolyChordSettings(nDims, nDerived)
    settings.root_dir = 'chains/polychord/'
    settings.file_root = root
    settings.nlive = nlive
    settings.do_clustering = True
    settings.read_resume = False

    def polychord_loglikelihood(theta):
        return loglikelihood(theta), []

    def polychord_dumper(live, dead, logweights, logZ, logZerr):
        dumper(live[:,:-1])

    pypolychord.run_polychord(polychord_loglikelihood, nDims, nDerived, settings, prior, polychord_dumper) 
