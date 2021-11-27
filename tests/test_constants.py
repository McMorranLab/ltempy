import ltempy as lp
from ltempy import constants as _

# %%
def test_set_units():
	assert(lp.np.isclose(_.c,299792458.0))
	lp.set_units(meter=2)
	assert(lp.np.isclose(_.c,599584916.0))
