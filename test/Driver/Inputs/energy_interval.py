"""In the deterministic mode a step that writes energies moves the particles
as a step that does not (#97, #102).

  energy_interval.py TEMPLATE MDIR WORK TARGET PRECISION VARIANT...

TEMPLATE is a control file of 20 steps with `energy_interval = 1` and the
placeholders ELECTROSTATICS, CONSTRAINTS, TARGET, and PRECISION. Each
VARIANT is ELECTROSTATICS:CONSTRAINTS (CUTOFF or PME, true or false). For
each, the run with a row at every step and the run with one at the end are
compared bit for bit in positions, velocities, and forces; this prints
"same state" or the largest difference of each.
"""
import pathlib
import subprocess
import sys

template, cli, work, target, precision = sys.argv[1:6]
text = pathlib.Path(template).read_text()
for variant in sys.argv[6:]:
    electrostatics, constraints = variant.split(":")
    folder = pathlib.Path(work) / f"{target}-{precision}-{electrostatics}-{constraints}"
    folder.mkdir(parents=True, exist_ok=True)
    every = (text.replace("ELECTROSTATICS", electrostatics)
             .replace("CONSTRAINTS", constraints).replace("TARGET", target)
             .replace("PRECISION", precision))
    once = every.replace("energy_interval = 1\n", "energy_interval = 20\n") \
                .replace("every", "once")
    (folder / "every.toml").write_text(every)
    (folder / "once.toml").write_text(once)
    printed = {}
    for name in ("every", "once"):
        subprocess.run([cli, "run", f"{name}.toml"], cwd=folder, check=True,
                       stdout=subprocess.DEVNULL)
        printed[name] = {
            field: subprocess.run([cli, "checkpoint", f"--print={field}",
                                   f"{name}.h5"], cwd=folder, check=True,
                                  stdout=subprocess.PIPE, text=True).stdout
            for field in ("positions", "velocities", "forces")}
    differences = []
    for field in ("positions", "velocities", "forces"):
        a, b = printed["every"][field], printed["once"][field]
        if a == b:
            continue
        largest = max(abs(float(x) - float(y))
                      for p, q in zip(a.splitlines(), b.splitlines())
                      for x, y in zip(p.split()[2:], q.split()[2:]))
        differences.append(f"{field} differ by {largest:.3e}")
    count = len(printed["every"]["positions"].splitlines())
    print(f"{target} {precision} {electrostatics} constraints={constraints}: "
          + ("; ".join(differences) if differences else f"same state, {count} particles"))
