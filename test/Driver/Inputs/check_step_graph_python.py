from pathlib import Path
import sys

import mdir

execution = mdir.Execution()
assert execution.step_graph == "LAUNCH"
for value in ("AUTO", "GRAPH", "LAUNCH"):
    execution.step_graph = value
    assert execution.step_graph == value

inputs = Path(sys.argv[1])
data = mdir.load_amber(str(inputs / "dipeptide.prmtop"), str(inputs / "dipeptide.inpcrd"))
execution.target = mdir.Target.CPU
execution.step_graph = "AUTO"
args = (data.make_system(), data.make_state(), mdir.Integrator(),
        mdir.Ensemble(), execution, mdir.Schedule())
plan = mdir.compile(*args).plan
assert plan["step_graph"] == "LAUNCH" and plan["step_graph_requested"] == "AUTO"
assert "GPU" in plan["step_graph_reason"]

execution.step_graph = "GRAPH"
try:
    mdir.compile(*args)
except mdir.UnsupportedError as err:
    assert "step_graph" in str(err) and "GPU" in str(err)
else:
    raise AssertionError("CPU execution accepted GRAPH")

# SPANS records the work of `mdir run` between host boundaries
# (D[step-no-host-wait]); a simulation of the Python interface refuses it.
execution.step_graph = "SPANS"
try:
    mdir.compile(*args)
except mdir.UnsupportedError as err:
    assert "SPANS" in str(err) and "Python" in str(err), err
else:
    raise AssertionError("a Python simulation accepted SPANS")
