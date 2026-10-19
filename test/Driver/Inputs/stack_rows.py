"""The stack of `mdir run` does not grow with the rows that a run writes
(#305).

    stack_rows.py mdir work target precision

The entry of a run returns when the run ends, and what it takes of the stack
in its loops over steps it keeps until then. A row that is allocated where
it is written (the observed columns, the coordinates and the terms of
pulling, dH/dlambda with the energies of the states) was taken again at
every row: 16 bytes for every two numbers of it, on the CPU, until the stack
of the process, 8 MiB by default, was full, and the run ended with a
segmentation fault.

Each case is two argon atoms on a spring, without a cell, with a row at
every step, run under a limit of the stack of LIMIT bytes for as many steps
as would take FACTOR times the limit if a row were taken at every step. The
run must end with the status 0 and with its last row in each file. A control
without rows other than the energies shows what the limit leaves free. The
lowered module is read as well: on the CPU no allocation of the stack is
outside the first block of its function (a block of a loop is not the
first); on a GPU the lowering releases the stack where an iteration ends
(`releaseStack`), so an allocation outside the first block stands with a
`stacksave`.
"""
import pathlib
import re
import resource
import subprocess
import sys

cli, work = sys.argv[1], pathlib.Path(sys.argv[2])
target, precision = sys.argv[3], sys.argv[4]
mode = f"{target} {precision.capitalize()}"
work.mkdir(parents=True, exist_ok=True)

LIMIT = 2 << 20  # bytes of the stack; a run of these programs uses about 0.25 MiB
FACTOR = 3

(work / "two.top").write_text("""[ defaults ]
1 2 no 1.0 1.0
[ atomtypes ]
AR 18 39.948 0.0 A 0.0 0.0
[ moleculetype ]
AR 1
[ atoms ]
1 AR 1 AR AR 1 0.0 39.948
[ system ]
two atoms on a spring
[ molecules ]
AR 2
""")
(work / "two.gro").write_text("""Two atoms on a spring
    2
    1AR      AR    1   1.000   1.000   1.000
    2AR      AR    2   1.500   1.100   0.900
   0.00000   0.00000   0.00000
""")

STATES = 11
LAMBDAS = ", ".join(f"{k / (STATES - 1):.1f}" for k in range(STATES))

# name: (the keys of [output], the terms, what follows them, the files, the
# bytes of the stack that a row took where it was allocated in the loop: 16
# for every two numbers of each buffer).
CASES = {
    "the energies alone": ("", "", "", [], 0),
    "observe, 6 columns": (
        'observables = "{name}.obs"',
        """[[energy.bond]]
name       = "flat"
expression = "k*(r - r0)^2 + a*r + b*r^2 + c*r^3"
particles  = [[1, 2]]
observe    = ["r0", "k", "a", "b", "c"]
k  = 5.0
r0 = 5.2
a  = 0.01
b  = 0.001
c  = 0.0001
""", "", ["obs"], 48),
    "pull, 2 terms over two centers": (
        'pull = "{name}.pull"',
        """[[energy.bond]]
name       = "steer"
expression = "k*(r - (r0 + v*t))^2"
groups     = [":1", ":2"]
k  = 5.0
r0 = 5.2
v  = 0.0
[[energy.bond]]
name       = "back"
expression = "k*(r - r0)^2"
groups     = [":2", ":1"]
k  = 1.0
r0 = 5.0
""", "", ["pull"], 64 + 80),
    f"free energy, {STATES} states": (
        'free_energy = "{name}.dhdl"',
        """[[energy.bond]]
name       = "alch"
expression = "k*(r - (r0 + lambda_pull))^2"
particles  = [[1, 2]]
k  = 5.0
r0 = 5.2
""", f"[free_energy]\nstate = 0\n[free_energy.lambdas]\npull = [{LAMBDAS}]\n", ["dhdl"],
        # The row of 1 + 11 numbers and the table of the 11 states.
        96 + 96),
}


def control(name, steps, output, terms, more):
    return f"""[input]
topology    = "two.top"
coordinates = "two.gro"
[output]
energy_interval = 1
energy = "{name}.dat"
{output.format(name=name)}
[energy]
cutoff            = 9.0
pairlist_distance = 10.0
electrostatics    = "CUTOFF"
{terms}
[dynamics]
integrator = "VELOCITY_VERLET"
time_step  = 0.001
steps      = {steps}
[ensemble]
ensemble    = "NVE"
temperature = 0.0
[boundary]
type = "NONE"
{more}
[execution]
target    = "{target}"
precision = "{precision}"
"""


def limit():
    resource.setrlimit(resource.RLIMIT_STACK, (LIMIT, LIMIT))


def last_step(path):
    rows = [line for line in path.read_text().splitlines() if line and not line.startswith("#")]
    return int(rows[-1].split()[0]) if rows else -1


def allocations(text):
    """The allocations of the stack of a lowered module outside the first
    block of their function, and the ops that save the stack."""
    outside, saves, block = 0, 0, 0
    for line in text.splitlines():
        if re.match(r"\s*llvm\.func ", line):
            block = 0
        elif re.match(r"\s*\^bb\d+", line):
            block += 1
        elif "llvm.intr.stacksave" in line:
            saves += 1
        elif re.search(r"\.alloca(?!\w)", line) and block > 0:
            outside += 1
    return outside, saves


failed = False
for index, (name, (output, terms, more, files, row)) in enumerate(CASES.items()):
    # As many steps as would take FACTOR times the limit; the control takes
    # those of the smallest row.
    steps = FACTOR * LIMIT // (row or 48)
    tag = f"case{index}"
    (work / f"{tag}.toml").write_text(control(tag, steps, output, terms, more))
    process = subprocess.run([cli, "run", f"{tag}.toml"], cwd=work, preexec_fn=limit,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ends = [last_step(work / f"{tag}.{suffix}") for suffix in ["dat"] + files]
    whole = process.returncode == 0 and all(end == steps for end in ends)
    failed |= not whole
    status = (f"killed by signal {-process.returncode}" if process.returncode < 0
              else f"status {process.returncode}")
    print(f"{mode}: {name}: {steps} rows under a stack of {LIMIT >> 20} MiB "
          f"({row} bytes a row taken in the loop would be {steps * row / LIMIT:.1f} of it): "
          f"{status}, the last rows at {', '.join(str(end) for end in ends)}: {whole}")

    lowered = subprocess.run([cli, "emit", "--stage=lowered", f"{tag}.toml"], cwd=work,
                             check=True, capture_output=True, text=True).stdout
    outside, saves = allocations(lowered)
    if target == "CPU":
        clean = outside == 0
        print(f"{mode}: {name}: the lowered module has {outside} allocations of the stack "
              f"outside the first block of a function: {clean}")
        # Not only read: the lowering refuses a module that would keep one
        # (#312).
        pipeline = subprocess.run([cli, "emit", "--stage=pipeline", f"{tag}.toml"], cwd=work,
                                  check=True, capture_output=True, text=True).stdout
        guarded = "hoist-static-allocas{refuse-rest=true}" in pipeline
        print(f"{mode}: {name}: the lowering refuses an allocation of the stack that it "
              f"cannot take once: {guarded}")
        clean &= guarded
    else:
        clean = outside == 0 or saves > 0
        print(f"{mode}: {name}: the lowered module has no allocation of the stack outside "
              f"the first block of a function, or releases the stack of an iteration: {clean}")
    failed |= not clean
if failed:
    sys.exit("stack rows FAILED")
print("stack rows passed")
