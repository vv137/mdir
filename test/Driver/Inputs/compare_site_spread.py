"""Compares the two forms of the spreading of the force on virtual sites
(D298): MDIR_SITE_SPREAD=written, the transpose of the
placement written out, and derived, the transpose that md-differentiate
derives from the placement.

    compare_site_spread.py mdir run.toml forces columns

Runs the control file in each form, with a checkpoint after its steps and
the energies of every step in a file, and prints the largest difference of
the forces of the checkpoints, relative to the largest force, and of each
column of the energies (the virial among them), relative to the largest
value of the column, then "spread ok" if they are within `forces` and
`columns`; the columns have the printed digits of the file.
"""
import os
import subprocess
import sys

mdir, control = sys.argv[1], sys.argv[2]
tolerance, printed = float(sys.argv[3]), float(sys.argv[4])
directory = os.path.dirname(os.path.abspath(control))
text = open(control).read()


def run(mode):
    name = os.path.join(directory, f"spread-{mode}.toml")
    output = (f'[output]\ncheckpoint = "spread-{mode}.h5"\n'
              f'checkpoint_interval = 1\nenergy = "spread-{mode}.energy"\n')
    open(name, "w").write(text.replace("[output]\n", output, 1))
    env = dict(os.environ, MDIR_SITE_SPREAD=mode)
    subprocess.run([mdir, "run", name], cwd=directory, env=env, check=True,
                   capture_output=True)
    printed = subprocess.run(
        [mdir, "checkpoint", f"spread-{mode}.h5", "--print=forces"],
        cwd=directory, env=env, check=True, capture_output=True,
        text=True).stdout
    forces = [float(v) for line in printed.splitlines()
              for v in line.split()[2:5]]
    rows = [[float(v) for v in line.split()]
            for line in open(os.path.join(directory, f"spread-{mode}.energy"))
            if not line.startswith("#")]
    return forces, rows


written, derived = run("written"), run("derived")
scale = max(abs(f) for f in written[0])
force = max(abs(a - b) for a, b in zip(*[w[0] for w in (written, derived)]))
print(f"forces: {len(written[0]) // 3} particles, largest difference "
      f"{force / scale:.1e} of the largest force")
worst = 0.0
columns = len(written[1][0])
for c in range(2, columns):
    top = max(abs(r[c]) for r in written[1]) or 1.0
    diff = max(abs(a[c] - b[c]) for a, b in zip(written[1], derived[1]))
    worst = max(worst, diff / top)
print(f"energy columns: {len(written[1])} rows, largest difference "
      f"{worst:.1e}")
ok = (force / scale <= tolerance and worst <= printed
      and len(written[1]) == len(derived[1]))
print("spread ok" if ok else "spread FAILED")
