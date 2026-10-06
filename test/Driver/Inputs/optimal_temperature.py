"""The temperatures of harmonic wells under velocity Verlet with a
thermostat (D[optimal-temperature]).

  optimal_temperature.py LOG N OMEGA_DT TARGET

LOG is the log of a run of N particles in the wells of brownian_wells.py,
each three oscillators of frequency omega, with the step omega dt. Along
an orbit of velocity Verlet, <K> = (1 - w/4) <U> and <K_half> = <U> with
w = (omega dt)^2, so the optimal estimate (K + 2 K_half) / 3 that the log
reports is (1 - w/12) <U>. The thermostat holds the kinetic energy of the
velocities of the steps, K, at TARGET. This prints the mean temperature of
K against TARGET, and the ratios of it and of the temperature of the
configurations, 2 <U> / (3 N k_B), to the log's temperature against these
closed forms, with N_f = 3N - 3 for the temperatures of the log.
"""
import sys

log, n, wdt, target = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), \
    float(sys.argv[4])
kB = 0.0083144626181532 / 4.184
nf = 3 * n - 3
rows = []
for line in open(log):
    words = line.split()
    if words[:2] == ["INFO:", "STEP"]:
        columns = words[1:]
    elif len(words) > 2 and words[0] == "INFO:" and words[1].isdigit():
        rows.append(dict(zip(columns, map(float, words[1:]))))
rows = rows[len(rows) // 5:]


def mean(key):
    return sum(row[key] for row in rows) / len(rows)


w = wdt * wdt
optimal = mean("TEMPERATURE")
full = 2 * mean("KINETIC_ENE") / (nf * kB)
configurations = 2 * mean("POTENTIAL_ENE") / (3 * n * kB) / optimal
expected_full = (1 - w / 4) / (1 - w / 12)
expected_configurations = nf / (3 * n) / (1 - w / 12)
print("temperature of the steps %.2f K, target %.2f: %s" % (
    full, target, "agrees" if abs(full - target) < 0.01 * target
    else "differs"))
print("full / optimal %.4f, closed form %.4f: %s" % (
    full / optimal, expected_full,
    "agrees" if abs(full / optimal - expected_full) < 1e-3 else "differs"))
print("configurations / optimal %.4f, closed form %.4f: %s" % (
    configurations, expected_configurations,
    "agrees" if abs(configurations - expected_configurations) < 0.01
    else "differs"))
