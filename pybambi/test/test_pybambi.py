import pytest
import pybambi

def test_run_pyBAMBI_inputs():

    with pytest.raises(NotImplementedError):
        pybambi.run_pyBAMBI(nested_sampler='NAN')



