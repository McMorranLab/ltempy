import ltempy as lp
import ltempy.constants as _

print("c = {} m / s".format(_.c))
print("hbar = {} kg m^2 / s^2".format(_.hbar))
lp.set_units(meter = 1e-3)
print("c = {} km / s".format(_.c))
print("hbar = {} kg km^2 / s^2".format(_.hbar))
