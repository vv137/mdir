"""A case of the corpus of check.py (#739): particle mesh Ewald with the
potential at the particles, the fourth result of `md.reciprocal`, which only
the derivative of the energy in the charges asks for (D231) and which no
control file reaches. The program is that of the dipeptide in water with
the charges tunable and their gradient asked for, as in
test/Driver/Inputs/python_tunable_gradient.py.

    pme-potential.py TARGET PRECISION STAGE

TARGET is CPU or GPU, PRECISION is SINGLE, MIXED, DOUBLE, or DEFAULT
(mixed), and STAGE is module (Program.ir) or lowered (Program.lowered_ir).
check.py runs it with the Python package of each build on PYTHONPATH. An
error is printed as its type and message alone, without a traceback, whose
paths would name the build.
"""
import os
import sys

import mdir

target, precision, stage = sys.argv[1:4]
inputs = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "..", "test", "Driver", "Inputs", "dipeptide")
try:
    loaded = mdir.load_amber(os.path.join(inputs, "dipeptide.prmtop"),
                             os.path.join(inputs, "dipeptide.inpcrd"))
    system, state = loaded.make_system(), loaded.make_state()
    system.electrostatics = mdir.Electrostatics.PME
    system.tunables = [mdir.Tunable("q", "charge")]
    system.tunable_gradient = True
    execution = mdir.Execution()
    execution.target = getattr(mdir.Target, target.upper())
    execution.precision = getattr(
        mdir.Precision, "Mixed" if precision == "DEFAULT" else precision.capitalize())
    program = mdir.compile(system, state, mdir.Integrator(), mdir.Ensemble(),
                           execution, mdir.Schedule())
    text = program.ir if stage == "module" else program.lowered_ir
except Exception as error:  # the same message in both builds is identical
    sys.stderr.write(f"{type(error).__name__}: {error}\n")
    sys.exit(1)
sys.stdout.write(text)
