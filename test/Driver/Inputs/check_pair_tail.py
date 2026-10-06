"""The tail of pair terms beyond the cutoff in the correction for the
dispersion (D209), on the mixture of
pair_tail_system.py, against an independent reference in pure Python.

With a uniform density beyond the cutoff r_c, each unordered pair that a
term counts adds (4 pi / V) times the integral of r^2 u(r) from r_c to
infinity to the energy, and -(4 pi / V) times that of r^3 u'(r) to the trace
of the virial, both times n^2 / (n (n - 1)), the factor of the topology's
correction (no pair of these molecules is excluded). The topology's own
correction is -(2 pi / 3V) n^2 <C6> / r_c^3 with W = 6 E. The integrals here
are composite Simpson sums in ln r out to 10^4 r_c, with u'(r) written out by
hand: neither the quadrature nor the integration by parts of MDIR.

    check_pair_tail.py directory tolerance-kcal/mol run...
    check_pair_tail.py directory tolerance-kcal/mol derivatives

Each run is a name whose log, <name>.log, the directory holds; the name
says what it has: plain (plain.top alone), fixed (fixed.top), nbfix
(plain.top and the A-B correction of sigma' = 3.7 Å, epsilon' = 0.5
kcal/mol), r8 (plain.top and -c8/r^8 + screened over A-B and A-A, see
below). Prints, for each run, the dispersion line and the trace of the
virial against the reference, and for the pairs (fixed, nbfix) the
difference of the total energy, the virial, and the pressure, which is the
tail of the repulsion of the correction alone. With `derivatives`, the rows
of [free_energy] and `observe` of lambda(-off).dhdl and observe(-off).obs
(see derivatives below)."""

import math
import sys

KCAL = 4.184
ATM = 69476.95 / 1.01325  # atm per kcal/mol/Å^3, the pressure of the log
SIGMA, EPSILON = 3.4, 0.24
SIG, EPS = 3.7, 0.5
C8, A, L = 2.0e4, 0.8, 3.0


def lj(s, e):
    return (lambda r: 4 * e * ((s / r) ** 12 - (s / r) ** 6),
            lambda r: 4 * e * (-12 * s ** 12 / r ** 13 + 6 * s ** 6 / r ** 7))


def tails(u, du, rc):
    """The integrals of r^2 u and r^3 u' from rc to infinity."""
    n, top = 200000, math.log(1.0e4)
    h = top / n
    e = w = 0.0
    for k in range(n + 1):
        r = rc * math.exp(k * h)
        weight = 1 if k in (0, n) else (4 if k % 2 else 2)
        e += weight * r ** 3 * u(r)
        w += weight * r ** 4 * du(r)
    return e * h / 3, w * h / 3


def read_log(path):
    terms, row, diagonal = {}, None, None
    lines = open(path).read().splitlines()
    for k, line in enumerate(lines):
        parts = line.split()
        if line.startswith("MDIR:   ") and len(parts) >= 3:
            try:
                terms[" ".join(parts[1:-1])] = float(parts[-1])
            except ValueError:
                pass
        if "diagonal of the virial" in line:
            diagonal = [float(x) for x in lines[k + 1].split()[1:4]]
        if line.startswith("INFO:") and row is None and parts[1] == "0":
            row = [float(x) for x in parts[1:]]
    return terms, row, diagonal


def read_columns(path):
    lines = open(path).read().splitlines()
    names = lines[0].lstrip("#").split()
    row = [float(x) for x in lines[2].split()]
    return dict(zip(names, row))


def derivatives(directory, tolerance, volume, na, nb, scale, rc):
    """The rows at step 0 of [free_energy] and `observe` with the
    correction on and off: they are of the shifted potential, so they differ
    by the tail and the estimate of the shift, (4 pi / 3V) r_c^3 u(r_c) for
    each pair (D[shifted-derivatives]), and their derivatives, written out
    from the closed forms of the integrals of powers."""
    factor = 4 * math.pi / volume * scale
    ok = True
    # -lambda_s c8 / r^8 over A-B and A-A, at lambda_s = 0.5 of (0, 0.5, 1):
    # the tail -c8 / (5 r_c^5) and the shift -c8 / (3 r_c^5) of a pair.
    tail = -factor * (na * nb + na * (na - 1) / 2) * C8 * (
        1 / (5 * rc ** 5) + 1 / (3 * rc ** 5))
    on = read_columns(f"{directory}/lambda.dhdl")
    off = read_columns(f"{directory}/lambda-off.dhdl")
    expected = {"dHdl.s": tail, "dU.0": -0.5 * tail, "dU.1": 0.0,
                "dU.2": 0.5 * tail}
    for name, value in expected.items():
        difference = on[name] - off[name] - value
        print(f"{name}: tail {value:.6f}, difference {difference:.1e}")
        ok &= abs(difference) <= tolerance

    def lj_tail(s, e):
        # The tail and the shift, r_c^3 u(r_c) / 3.
        return 4 * e * (s ** 12 / (9 * rc ** 9) - s ** 6 / (3 * rc ** 3)
                        + s ** 12 / (3 * rc ** 9) - s ** 6 / (3 * rc ** 3))

    pairs = factor * na * nb
    on = read_columns(f"{directory}/observe.obs")
    off = read_columns(f"{directory}/observe-off.obs")
    expected = {
        "nbfix.energy": pairs * (lj_tail(SIG, EPS) - lj_tail(SIGMA, EPSILON)),
        "nbfix.d_sig": pairs * 4 * EPS * (12 * SIG ** 11 / (9 * rc ** 9)
                                          - 6 * SIG ** 5 / (3 * rc ** 3)
                                          + 12 * SIG ** 11 / (3 * rc ** 9)
                                          - 6 * SIG ** 5 / (3 * rc ** 3)),
        "nbfix.d_eps": pairs * lj_tail(SIG, 1.0),
    }
    for name, value in expected.items():
        difference = on[name] - off[name] - value
        print(f"{name}: tail {value:.6f}, difference {difference:.1e}")
        ok &= abs(difference) <= tolerance
    print("derivatives: " + ("ok" if ok else "FAILED"))


def main():
    directory, tolerance = sys.argv[1], float(sys.argv[2])
    names = sys.argv[3:]
    gro = open(f"{directory}/system.gro").read().splitlines()
    count = int(gro[1])
    kinds = [line[5:10].strip() for line in gro[2:2 + count]]
    edge = float(gro[2 + count].split()[0]) * 10
    volume = edge ** 3
    na, nb = kinds.count("A"), kinds.count("B")
    n = na + nb
    scale = n * n / (n * (n - 1))
    rc = 12.0
    if names == ["derivatives"]:
        derivatives(directory, tolerance, volume, na, nb, scale, rc)
        return

    def topology(c6ab):
        c6 = 4 * EPSILON * SIGMA ** 6
        total = (na * (na - 1) + nb * (nb - 1)) * c6 + 2 * na * nb * c6ab
        e = -2 * math.pi / (3 * volume) * n * n * total / (n * (n - 1)) / rc ** 3
        # The shift of -C6/r^6 alone, -(4 pi / 3V) r_c^3 C6 / r_c^6 a pair,
        # equals its tail.
        return e, 6 * e, e

    def tail(u, du, pairs):
        i, j = tails(u, du, rc)
        factor = 4 * math.pi / volume * scale * pairs
        return factor * i, -factor * j, factor * rc ** 3 * u(rc) / 3

    plain = topology(4 * EPSILON * SIGMA ** 6)
    fixed = topology(4 * EPS * SIG ** 6)
    u1, du1 = lj(SIG, EPS)
    u0, du0 = lj(SIGMA, EPSILON)
    nbfix = tail(lambda r: u1(r) - u0(r), lambda r: du1(r) - du0(r), na * nb)
    # -c8/r^8 over A-B and A-A, and -a exp(-r/l)/r^4 over the same pairs.
    r8 = tail(lambda r: -C8 / r ** 8 - A * math.exp(-r / L) / r ** 4,
              lambda r: 8 * C8 / r ** 9
              + A * math.exp(-r / L) * (1 / (L * r ** 4) + 4 / r ** 5),
              na * nb + na * (na - 1) / 2)
    reference = {
        "plain": plain,
        "fixed": fixed,
        "nbfix": tuple(a + b for a, b in zip(plain, nbfix)),
        "r8": tuple(a + b for a, b in zip(plain, r8)),
    }
    logs = {}
    ok = True
    for name in names:
        terms, row, diagonal = read_log(f"{directory}/{name}.log")
        logs[name] = (terms, row, diagonal)
        # Under POTENTIAL_SHIFT (a name ending in -shift) the correction adds
        # the estimate of the shift (D[shifted-derivatives]).
        e, w, shift = reference[name.split("-")[0]]
        if name.endswith("-shift"):
            e += shift
        de = terms["dispersion"] - e
        print(f"{name}: dispersion {terms['dispersion']:.6f} reference {e:.6f} "
              f"difference {de:.1e}")
        ok &= abs(de) <= max(tolerance, 1.5e-6)
    # The NBFIX as a correction term against [ nonbond_params ]: they differ
    # by the tail of the repulsion of the correction, which the topology's
    # correction leaves out.
    for name in names:
        if not name.startswith("nbfix"):
            continue
        other = "fixed" + name[len("nbfix"):]
        if other not in logs:
            continue
        (t1, r1, d1), (t0, r0, d0) = logs[name], logs[other]
        c12 = 4 * EPS * SIG ** 12 - 4 * EPSILON * SIGMA ** 12
        e12 = 4 * math.pi / volume * scale * na * nb * c12 / (9 * rc ** 9)
        # And under POTENTIAL_SHIFT the estimate of the shift of the
        # repulsion, 3 times its tail, which has no virial.
        e12s = e12 * (4 if name.endswith("-shift") else 1)
        de = t1["total"] - t0["total"] - e12s
        dw = [a - b - 4 * e12 for a, b in zip(d1, d0)]
        dp = r1[-1] - r0[-1] - 12 * e12 / (3 * volume) * ATM
        print(f"{name} - {other}: repulsion tail {e12s:.6e}; total {de:.1e}, "
              f"virial {max(abs(x) for x in dw):.1e}, pressure {dp:.1e} atm")
        ok &= abs(de) <= tolerance and max(abs(x) for x in dw) <= tolerance
        ok &= abs(dp) <= max(1.5e-4, tolerance / (3 * volume) * ATM * 12)
    print("tails: " + ("ok" if ok else "FAILED"))


if __name__ == "__main__":
    main()
