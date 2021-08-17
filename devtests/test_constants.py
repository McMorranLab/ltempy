import ltempy as lp
import ltempy.constants as _

print("c = {} m / s".format(_.c))
print("hbar = {}".format(_.hbar))
lp.set_units(meter = 1e-3)
print("c = {} km / s".format(_.c))
print("With km as the base unit of length, hbar = {}".format(_.hbar))
