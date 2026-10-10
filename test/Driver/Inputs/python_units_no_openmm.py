"""The Python model without OpenMM (D200): `openmm` is hidden,
MDIR imports and its plain setters work without importing it, and an
object that looks like a quantity raises InputError."""
import sys

# A None entry makes `import openmm` and `import openmm.unit` fail.
sys.modules["openmm"] = None
sys.modules["openmm.unit"] = None

import numpy as np
import mdir

root = sys.argv[1]
loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
system, state = loaded.make_system(), loaded.make_state()
# The defaults of before D[python-defaults], with which this was written.
system.truncation = mdir.Truncation.Switch
system.switch_distance = 1.0
system.cutoff = 0.8
system.pairlist_distance = 9  # an int is a plain number too
ensemble, integrator = mdir.Ensemble(), mdir.Integrator()
ensemble.pressure = np.float64(1.01325)
integrator.timestep = 0.002
state.positions = np.asarray(state.positions).copy()
system.restraints = [mdir.Restraint("@CA", 4184.0)]
system.restraint_reference = np.asarray(state.positions)
assert system.cutoff == 0.8 and system.pairlist_distance == 9.0
assert system.restraints[0].force_constant == 4184.0


class LooksLikeAQuantity:
    def value_in_unit_system(self, system):
        return 1.0

    def value_in_unit(self, unit):
        return 1.0


try:
    system.cutoff = LooksLikeAQuantity()
except mdir.InputError as exc:
    assert "System.cutoff: a unit quantity needs openmm.unit" in str(exc), str(exc)
else:
    raise AssertionError("expected InputError")
assert system.cutoff == 0.8
assert sys.modules["openmm"] is None and "openmm.unit" in sys.modules
print("plain setters without openmm passed")
