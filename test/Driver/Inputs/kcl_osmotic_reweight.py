"""The mechanics of the reweighting of examples/kcl-osmotic (#140): the
columns nbfix.d_x12 and nbfix.d_x6 of `observe`, and the exact reweighting
in sigma' that they give, against the energies of the same states at
another sigma'.

    kcl_osmotic_reweight.py write DIR EXAMPLE   (after pair_tail_system.py)
    kcl_osmotic_reweight.py frame DIR EXAMPLE   (after the run)
    kcl_osmotic_reweight.py check DIR EXAMPLE

`write` makes run.toml: the A-B pairs of the mixture take the correction
x12 c12/r^12 - x6 c6/r^6 of the example at sigma_hat = sigma0 (x12 = x6 = 0),
observed in x12 and x6, with a state of [free_energy] at
sigma1 = sigma0 + 0.02 A (lambda_s = 1), so that every row of the run has
U(sigma1) - U(sigma_hat) of its own state; walls on z and the count of a
slab as in the example. The run writes its last frame to run.dcd; `frame`
writes it as last.gro, and point-hat.toml and point-new.toml evaluate it at
sigma_hat and at sigma1.

`check` compares, at every row, dU.1 with dx12 A12 + dx6 A6 of the columns;
the reweighted mean of a wall's column and N_eff/N from the columns (the
example's osmotic.py) with those from dU.1; and the difference of the
energies of the two single points with dx12 A12 + dx6 A6 of the columns of
the one at sigma_hat.
"""
import sys

import numpy as np

mode, out, example = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, example)
import osmotic  # noqa: E402

S1 = osmotic.SIG0 + 0.02
X12, X6 = osmotic.x_of(S1)

TERMS = f"""
[[energy.pair]]
name       = "nbfix"
groups     = [":A", ":B"]
expression = "(x12 + lambda_s*d12)*c12/r^12 - (x6 + lambda_s*d6)*c6/r^6"
observe    = ["x12", "x6"]
c12 = {osmotic.C12!r}
c6  = {osmotic.C6!r}
x12 = @X12@
x6  = @X6@
d12 = {X12!r}
d6  = {X6!r}

[[energy.external]]
name       = "lower"
selection  = ":A,B"
scaling    = "NONE"
expression = "0.5*k*max(0, z0 - z)^2"
observe    = ["z0"]
k  = 10.0
z0 = 9.0

[[energy.external]]
name       = "upper"
selection  = ":A,B"
scaling    = "NONE"
expression = "0.5*k*max(0, z - z0)^2"
observe    = ["z0"]
k  = 10.0
z0 = 21.0

[[energy.external]]
name       = "slab"
selection  = ":A,B"
scaling    = "NONE"
expression = "s*step(zw - abs(z - zc))"
observe    = ["s"]
s  = 0.0
zc = 15.0
zw = 3.0

[free_energy.lambdas]
s = [0.0, 1.0]

[energy]
cutoff         = 9.0
electrostatics = "PME"
lennard_jones_modifier = "POTENTIAL_SHIFT"
dispersion_correction  = "ENERGY_PRESSURE"

[boundary]
type = "PERIODIC"

[execution]
target    = "CPU"
precision = "DOUBLE"
threads   = 2
"""

RUN = """[input]
topology    = "plain.top"
coordinates = "system.gro"
[output]
observables         = "run.obs"
free_energy         = "run.dhdl"
trajectory          = "run.dcd"
energy_interval     = 10
trajectory_interval = 2000
[dynamics]
integrator = "VELOCITY_VERLET"
time_step  = 0.002
steps      = 2000
seed       = 140
[ensemble]
ensemble    = "NVT"
temperature = 300.0
[thermostat]
method   = "LANGEVIN"
friction = 5.0
[free_energy]
state = 0
"""

POINT = """[input]
topology    = "plain.top"
coordinates = "last.gro"
[output]
observables     = "NAME.obs"
energy_interval = 1
[dynamics]
integrator = "VELOCITY_VERLET"
time_step  = 0.002
steps      = 1
[ensemble]
ensemble    = "NVE"
temperature = 0.0
[free_energy]
state = 0
"""


def last_frame(path, natoms):
    """x, y, z (A) of the last frame of a CHARMM DCD: its last three records."""
    data, records, i = open(path, "rb").read(), [], 0
    while i < len(data):
        size = int(np.frombuffer(data[i:i + 4], "<i4")[0])
        records.append(data[i + 4:i + 4 + size])
        i += size + 8
    xyz = [np.frombuffer(r, "<f4") for r in records if len(r) == 4 * natoms][-3:]
    return np.stack(xyz, axis=1).astype(float)


def table(path):
    with open(path) as f:
        names = f.readline().lstrip("#").split()
    a = np.loadtxt(path, comments="#", ndmin=2)
    return {n: a[:, i] for i, n in enumerate(names)}


def total(log):
    for line in open(log):
        if line.startswith("MDIR:   total"):
            return float(line.split()[-1])
    raise SystemExit(f"no total in {log}")


if mode == "write":
    terms = TERMS.replace("@X12@", "0.0").replace("@X6@", "0.0")
    open(f"{out}/run.toml", "w").write(RUN + terms)
    open(f"{out}/point-hat.toml", "w").write(POINT.replace("NAME", "point-hat") + terms)
    new = TERMS.replace("@X12@", repr(X12)).replace("@X6@", repr(X6))
    open(f"{out}/point-new.toml", "w").write(POINT.replace("NAME", "point-new") + new)
    sys.exit(0)

if mode == "frame":
    gro = open(f"{out}/system.gro").read().splitlines()
    natoms = int(gro[1])
    x = last_frame(f"{out}/run.dcd", natoms) / 10.0
    lines = gro[:2] + [g[:20] + "%8.3f%8.3f%8.3f" % tuple(p) for g, p in zip(gro[2:2 + natoms], x)] + [gro[2 + natoms]]
    open(f"{out}/last.gro", "w").write("\n".join(lines) + "\n")
    sys.exit(0)

obs, dhdl = table(f"{out}/run.obs"), table(f"{out}/run.dhdl")
assert np.array_equal(obs["step"], dhdl["step"]), "the rows of the two files differ"
n = len(obs["step"])
linear = X12 * obs["nbfix.d_x12"] + X6 * obs["nbfix.d_x6"]
gap = np.max(np.abs(dhdl["dU.1"] - linear))
print(f"rows {n}; |dU| up to {np.max(np.abs(dhdl['dU.1'])):.3f} kcal/mol; "
      f"largest |dU.1 - (dx12 A12 + dx6 A6)| {gap:.1e}")
print("columns:", "ok" if gap < 2e-5 and np.max(np.abs(dhdl["dU.1"])) > 0.1 else "FAIL")

o = obs["lower.d_z0"] - obs["upper.d_z0"]
w_col = osmotic.normalized(osmotic.log_weights(obs["nbfix.d_x12"], obs["nbfix.d_x6"], osmotic.SIG0, S1))
w_lam = osmotic.normalized(-dhdl["dU.1"] / (osmotic.KB * 300.0))
m_col, m_lam = float(w_col @ o), float(w_lam @ o)
n_col, n_lam = osmotic.kish(w_col), osmotic.kish(w_lam)
print(f"reweighted <O>: {m_col:.6f} from the columns, {m_lam:.6f} from dU.1; "
      f"N_eff/N {n_col:.4f} and {n_lam:.4f}")
print("reweighting:", "ok" if abs(m_col - m_lam) <= 1e-6 * max(1.0, abs(m_lam))
      and abs(n_col - n_lam) < 1e-6 else "FAIL")

hat = table(f"{out}/point-hat.obs")
du = total(f"{out}/point-new.log") - total(f"{out}/point-hat.log")
lin = X12 * hat["nbfix.d_x12"][0] + X6 * hat["nbfix.d_x6"][0]
print(f"single points of the last frame: U(sigma1) - U(sigma_hat) = {du:.6f}, "
      f"dx12 A12 + dx6 A6 = {lin:.6f} kcal/mol")
print("single points:", "ok" if abs(du - lin) < 2e-5 and abs(du) > 0.1 else "FAIL")
