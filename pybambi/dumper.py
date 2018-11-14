
def dumper(live):
    """ Dumper function giving access to the live and dead points """

    params = live[:,:-1]
    loglikes = live[:,-1]

    print("-----------------------------")
    print("Call neural network code here")
    print(loglikes)
    print("-----------------------------")



