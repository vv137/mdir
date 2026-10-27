"""The choices of a model term are read without regard to case, as those
of [[energy.model]] are (#551): `backend` and `stage_field_precision` of
`mdir.ModelTerm` reach the checks of the control in upper case.

usage: model_choice_case.py DIR"""
import os
import sys

import mdir

directory = sys.argv[1]
loaded = mdir.load_gromacs(os.path.join(directory, "system.top"),
                           os.path.join(directory, "system.gro"))


def refusal(target, precision, **term):
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.3, 0.35
    system.models = [mdir.ModelTerm("allegro", os.path.join(directory, "absent.pt"), **term)]
    execution = mdir.Execution()
    execution.target = target
    execution.precision = precision
    try:
        mdir.compile(system, state, mdir.Integrator(), mdir.Ensemble(), execution,
                     mdir.Schedule())
    except Exception as error:
        return str(error)
    return "accepted"


# A model in stages runs on a GPU only: the refusal of the CPU target names
# the model in stages, which "stages" and "Stages" select as "STAGES" does.
for backend in ("STAGES", "stages", "Stages"):
    print(backend + ":", refusal(mdir.Target.CPU, mdir.Precision.Double, backend=backend))
# A plugin's backend, in any case, takes the checks of a plugin.
for backend in ("METATOMIC", "metatomic", "Aoti"):
    print(backend + ":", refusal(mdir.Target.CPU, mdir.Precision.Double, backend=backend))
print("TORCHSCRIPT:", refusal(mdir.Target.CPU, mdir.Precision.Double, backend="TORCHSCRIPT"))
# "double" is "DOUBLE", which the checks of a model in stages take; another
# value is refused there.
for value in ("DOUBLE", "double", "SINGLE"):
    print(value + ":", refusal(mdir.Target.GPU, mdir.Precision.Mixed, backend="stages",
                               stage_field_precision=value))
