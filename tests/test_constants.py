import ltempy as wt
from ltempy import constants as _

# %%
def test_setUnits():
	assert(wt.np.isclose(_.c,299792458.0))
	wt.set_units(meter=2)
	assert(wt.np.isclose(_.c,599584916.0))
