"""D449: Execution.term_streams in Python: the values, the
plan's record, and the refusal of an unknown value."""
from pathlib import Path
import sys

import mdir

execution = mdir.Execution()
assert execution.term_streams == "SINGLE"
for value in ("RECIPROCAL", "TERMS", "SINGLE"):
    execution.term_streams = value
    assert execution.term_streams == value

inputs = Path(sys.argv[1])
data = mdir.load_amber(str(inputs / "dipeptide.prmtop"), str(inputs / "dipeptide.inpcrd"))
execution.target = mdir.Target.CPU
execution.term_streams = "TERMS"
args = (data.make_system(), data.make_state(), mdir.Integrator(),
        mdir.Ensemble(), execution, mdir.Schedule())
plan = mdir.compile(*args).plan
assert plan["term_streams"] == "SINGLE" and plan["term_streams_requested"] == "TERMS", plan

execution.term_streams = "BOTH"
try:
    mdir.compile(*args)
except mdir.InputError as err:
    assert "term_streams" in str(err), err
else:
    raise AssertionError("an unknown term_streams was accepted")
print("term_streams in Python: ok")
