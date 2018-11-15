def dumper(live):
    """ Dumper function giving access to the live points
    
    Parameters
    ----------
    live: numpy.array
          live parameters and loglikelihoods, `shape=(nlive,nDims+1)` 
    """

    params = live[:, :-1]
    loglikes = live[:, -1]

    print("-----------------------------")
    print("Call neural network code here")
    print(loglikes)
    print("-----------------------------")
