"""K_half of the log against the half steps of the trajectory
(D[optimal-temperature]).

  chord_half_steps.py LOG DT X1 X2 ... XN

LOG has a row at every step of a run of velocity Verlet; X1..XN are the
positions (`mdir checkpoint --print=positions`) of runs of 1..N steps from
the same start. Velocity Verlet drifts by dt v_{n+1/2} with the velocities
after the constraints of the positions, so v_{n+1/2} = (x_{n+1} - x_n) / dt,
the chord. The log's row n gives K and the optimal estimate
K_T = (K + 2 K_half) / 3, so K_half = K + 3 (K_T - K) / 2; this compares
it with [K(v_{n-1/2}) + K(v_{n+1/2})] / 2 at the steps 2..N-1. N_f comes
from the row of step 0, where K_T = K.
"""
import sys

log, dt, paths = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
kB = 0.0083144626181532 / 4.184
rows = {}
for line in open(log):
    words = line.split()
    if words[:2] == ["INFO:", "STEP"]:
        columns = words[1:]
    elif len(words) > 2 and words[0] == "INFO:" and words[1].isdigit():
        row = dict(zip(columns, map(float, words[1:])))
        rows[int(row["STEP"])] = row
freedom = 2 * rows[0]["KINETIC_ENE"] / (kB * rows[0]["TEMPERATURE"])
masses, positions = [], []
for path in paths:
    masses, x = [], []
    for line in open(path):
        words = line.split()
        masses.append(float(words[1]))
        x.append([float(v) for v in words[2:5]])
    positions.append(x)


def chord(a, b):
    """The kinetic energy in kcal/mol of the chord from run a to run b."""
    total = 0.0
    for m, p, q in zip(masses, positions[a], positions[b]):
        for c in range(3):
            d = q[c] - p[c]
            assert abs(d) < 0.5, "a particle crossed the cell"
            total += 0.5 * m * (d / dt) ** 2
    return total / 4.184


largest = 0.0
for n in range(2, len(paths)):
    row = rows[n]
    k = row["KINETIC_ENE"]
    optimal = row["TEMPERATURE"] * freedom * kB / 2
    logged = k + 1.5 * (optimal - k)
    # Run n - 1 ends at step n (runs of 1..N steps).
    actual = 0.5 * (chord(n - 2, n - 1) + chord(n - 1, n))
    largest = max(largest, abs(logged - actual))
    print("step %d: K_half %.4f, from the chords %.4f, K %.4f" %
          (n, logged, actual, k))
# The log has four decimals: K_T is known to 1e-4 K, times N_f k_B / 2.
tolerance = 3 * 1e-4 * freedom * kB / 2
print("largest difference %.5f kcal/mol: %s" %
      (largest, "agrees" if largest < tolerance else "differs"))
