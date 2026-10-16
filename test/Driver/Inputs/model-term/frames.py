"""A frame evaluator refuses a model term (D[metatomic-plugin]): its forces
and virial come from evaluations that do not take the model. It needs
tunables, which a model term refuses first; the builder refuses the frame
evaluator as well.

usage: frames.py DIR ARTIFACT"""
import os
import sys

import torch  # noqa: F401

import mdir

directory, artifact = sys.argv[1:3]
loaded = mdir.load_gromacs(os.path.join(directory, "system.top"),
                           os.path.join(directory, "system.gro"))
system, state = loaded.make_system(), loaded.make_state()
system.cutoff, system.pairlist_distance = 0.2, 0.25
system.models = [mdir.ModelTerm("allegro", artifact)]
execution = mdir.Execution()
execution.target = mdir.Target.GPU
program = mdir.compile(system, state, mdir.Integrator(), mdir.Ensemble(), execution,
                       mdir.Schedule())
try:
    mdir.FrameEvaluator(program)
    print("accepted")
except Exception as error:  # the kind is that of the compile's failure
    print("refused:", error)
