"""The automatic grid of the prolate kernel with the B-spline window
(D[prolate-bspline-grid], issue #985) on 400 charges at uncorrelated places,
where the estimate of the grid is the expected error: the error of the
reciprocal force, the forces less those of a grid three times finer, over
the reciprocal force, the forces of that grid less the direct sum of the
constrained density by NumPy. At each tolerance the automatic grid is finer
than that of eq. 15 of Liang et al. (2026), whose error is above the
tolerance; the error is 1.1 to 1.55 times the estimate E_B that the
manifest of `mdir run` records for that grid (measured: 1.29 at 1e-3 and
1.35 at 1e-4, above the 1.10 to 1.19 of 1200 charges, since the excess, the
force of a charge on itself, falls with their number); and the error lies
between 0.7 and 1.3 times the tolerance (measured 0.77 and 1.12). An
estimate twice too high fails both: the grid where the doubled estimate
meets the tolerance has E_B at most half of it, an error at most about 0.68
times it, and the error over the doubled estimate is near 0.67. With the
estimate doubled in the driver, the grids were 28^3 and 60^3, the error
0.52 and 0.58 to 0.59 times the tolerance, and 0.66 and 0.69 to 0.70 times
the estimate logged; the bounds of before, 0.4 and 1.3 times the tolerance,
passed it. The margin of the bound 0.7 at 1e-3 is small (0.68 to 0.77)."""
import json
import math
from pathlib import Path
import subprocess
import sys

import mdir
import numpy as np
from numpy.polynomial import legendre

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from mdir_physical_constants import COULOMB_NATIVE

target, work, executable = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
work.mkdir(parents=True, exist_ok=True)
count, edge, cutoff = 400, 3.0, 0.9
rng = np.random.default_rng(985)
q = np.where(np.arange(count) % 2 == 0, 1.0, -1.0)
gro = work / "ions.gro"
gro.write_text(f"ions\n{count}\n" + "".join(
    f"{(i + 1) % 100000:5d}ION     X{(i + 1) % 100000:5d}{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}\n"
    for i, p in enumerate(rng.uniform(0, edge, size=(count, 3)))) + f"{edge:10.5f}" * 3 + "\n")
x = np.array([[float(l[20:28]), float(l[28:36]), float(l[36:44])]
              for l in gro.read_text().splitlines()[2:2 + count]])
top = work / "ions.top"
top.write_text("[ defaults ]\n1 2 no 1 1\n[ atomtypes ]\nX 1 1.0 0 A 0.3 0\n"
               "[ moleculetype ]\nions 0\n[ atoms ]\n" +
               "".join(f"{i} X 1 ION X {i} {c} 1.0\n" for i, c in enumerate(q, 1)) +
               "[ system ]\nions\n[ molecules ]\nions 1\n")


def density(c, continuity=2, terms=8):
    """chi, int_0^1 chi = 1, as prolate_continuity_forces.py builds it."""
    u, w = legendre.leggauss(128)
    n = np.arange(0, 2 * terms, 2)
    basis = legendre.legvander(u, 2 * terms - 2)[:, ::2] * np.sqrt(2 * n + 1)
    constraints = np.array([np.ones(terms), n * (n + 1) / 2,
                            (n - 1) * n * (n + 1) * (n + 2) / 8])
    z = np.linalg.svd(constraints[:continuity] * np.sqrt(2 * n + 1),
                      full_matrices=True)[2][continuity:].T
    projected = (w[:, None] * basis) @ z
    kernel = np.sinc(c * (u[:, None] - u) / np.pi)
    _, vectors = np.linalg.eigh(projected.T @ kernel @ projected)
    coefficients = np.zeros(2 * terms - 1)
    coefficients[::2] = (z @ vectors[:, -1]) * np.sqrt(2 * n + 1)
    chi = legendre.Legendre(coefficients)
    return chi / (chi.integ()(1.0) - chi.integ()(0.0))


def direct(c):
    """f q q [(1 - Phi)/r^3 + chi/(r_c r^2)] d within the cutoff."""
    chi = density(c)
    phi = chi.integ(lbnd=0)
    force = np.zeros((count, 3))
    for i in range(count - 1):
        d = x[i] - x[i + 1:]
        d -= edge * np.rint(d / edge)
        r = np.linalg.norm(d, axis=1)
        inside = r < cutoff
        d, r, t = d[inside], r[inside], r[inside] / cutoff
        pair = (q[i] * q[i + 1:][inside] *
                ((1 - phi(t)) / r ** 3 + chi(t) / (cutoff * r ** 2)))[:, None] * d
        force[i] += pair.sum(0)
        np.subtract.at(force, np.arange(i + 1, count)[inside], pair)
    return COULOMB_NATIVE * force


loaded = mdir.load_gromacs(str(top), str(gro))
system, state = loaded.make_system(), loaded.make_state()
system.electrostatics = mdir.Electrostatics.PME
system.cutoff, system.pairlist_distance = cutoff, 1.0
system.pme_kernel = mdir.PMEKernel.Prolate
system.pme_continuity = 2
state.velocities = np.zeros_like(x)


def run(precision, grid=None):
    system.pme_grid = grid or [0, 0, 0]
    execution = mdir.Execution()
    execution.target = getattr(mdir.Target, target)
    execution.precision = getattr(mdir.Precision, precision)
    integrator = mdir.Integrator()
    integrator.method = mdir.IntegratorMethod.VelocityVerlet
    integrator.timestep = 1e-12
    program = mdir.compile(system, state, integrator, mdir.Ensemble(), execution,
                           mdir.Schedule())
    simulation = mdir.Simulation(program)
    simulation.run(1)
    return list(program.plan["pme_grid"]), np.asarray(simulation.state().forces, dtype=float)


def manifest(tolerance):
    """The grid and E_B that `mdir run` records for this system."""
    name = f"estimate-{tolerance:g}"
    (work / f"{name}.toml").write_text(f"""[input]
topology = "ions.top"
coordinates = "ions.gro"
[output]
energy_interval = 0
manifest = "{name}.jsonl"
[energy]
cutoff = {10 * cutoff}
pairlist_distance = 10.0
electrostatics = "PME"
[pme]
kernel = "PROLATE"
continuity = 2
tolerance = {tolerance:g}
[dynamics]
integrator = "VELOCITY_VERLET"
time_step = 0.0005
steps = 0
[ensemble]
ensemble = "NVE"
temperature = 300.0
[boundary]
type = "PERIODIC"
[execution]
target = "{target}"
""")
    done = subprocess.run([executable, "run", f"{name}.toml"], cwd=work, text=True,
                          capture_output=True)
    assert done.returncode == 0, done.stdout + done.stderr
    start = json.loads((work / f"{name}.jsonl").read_text().splitlines()[0])
    pme = start["effective"]["pme"]
    assert pme["grid_source"] == "automatic" and pme["aliasing_bound"] == tolerance, pme
    return pme["grid"], pme["aliasing_estimate"]


for tolerance, bandwidth in [(1e-3, 9.5392), (1e-4, 12.024)]:
    system.pme_tolerance = tolerance
    # Eq. 15: the grid before the refinement of the B-spline window.
    start = math.ceil(edge / (math.pi * cutoff / bandwidth) - 1e-9)
    start += start % 2
    chosen, _ = run("Double")
    fine_grid = [3 * k for k in chosen]
    _, fine = run("Double", fine_grid)
    reciprocal = fine - direct(bandwidth)
    scale = np.sqrt(np.mean(reciprocal ** 2))
    _, coarse = run("Double", [start] * 3)
    before = np.sqrt(np.mean((coarse - fine) ** 2)) / scale
    print(f"{target} tolerance {tolerance:g}: the grid of eq. 15 {start}^3, error "
          f"{before:.4g}; the automatic grid {chosen[0]}x{chosen[1]}x{chosen[2]}", flush=True)
    assert before > tolerance, before
    assert all(k > start for k in chosen), chosen
    logged, estimate = manifest(tolerance)
    assert logged == chosen, (logged, chosen)
    for precision in ("Mixed", "Double"):
        grid, forces = run(precision)
        assert grid == chosen, (grid, chosen)
        error = np.sqrt(np.mean((forces - fine) ** 2)) / scale
        print(f"{target} {precision} tolerance {tolerance:g}: reciprocal force error "
              f"{error:.4g} of its RMS {scale:.6g} kJ/mol/nm against the grid "
              f"{fine_grid[0]}^3, {error / tolerance:.3f} of the tolerance, bounds 0.7 and 1.3; "
              f"{error / estimate:.3f} of the estimate {estimate:.4g}, bounds 1.1 and 1.55",
              flush=True)
        assert 1.1 * estimate < error < 1.55 * estimate, (error, estimate)
        assert 0.7 * tolerance < error < 1.3 * tolerance, error
print("passed")
