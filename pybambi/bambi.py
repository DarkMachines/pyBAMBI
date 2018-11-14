from pybambi.dumper import dumper


def run_pyBAMBI(loglikelihood, prior, nDims, **kwargs):
    """ run pyBAMBI 

    Parameters
    ----------
    nested_sampler: str
        Choice of nested sampler. Must be in ['multinest', 'polychord'].
        Default 'polychord'.

    nlive: int
        Number of live points.
        Default nDims*25

    root: str
        root of filename.
        Default 'test'

    
    """
    nested_sampler = kwargs.pop('nested_sampler', 'polychord')
    nlive = kwargs.pop('nlive', nDims*25)
    root = kwargs.pop('root', 'test')

    if nested_sampler not in ['multinest', 'polychord']:
        raise NotImplementedError('nested sampler %s is not implemented' % nested_sampler)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    if nested_sampler == 'polychord':
        from pybambi.polychord import run_polychord
        run_polychord(loglikelihood, prior, dumper, nDims, nlive, root)

    elif nested_sampler == 'multinest': 
        from pybambi.multinest import run_multinest
        run_multinest(loglikelihood, prior, dumper, nDims, nlive, root)
    

