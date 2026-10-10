"""Tunable parameters of a Python model (D213,
docs/python-tunable.md).

Usage: python_tunable.py ROOT SCENARIO [TARGET PRECISION | CLI WORK]

Scenarios:
  declarations  the declarations, their refusals, and the arrays maps are
                built from (CPU, no run)
  updates       an update during a run against a simulation compiled with
                the new values from the same state (velocity Verlet, NVE),
                bit for bit; against a model without tunables; the refused
                updates, the versions, and the energy file
  integrators   leapfrog keeps its velocities through an update; an update
                before the first run of NPT against a compile with the new
                values, bit for bit
  pairs         the table tunable by pairs of types (#160): an update off
                the combining rule against a compile with the new values,
                bit for bit; the change of the energy, pair and per-type
                tunables together, and the correction for the dispersion
                against NumPy
  oracle        the energies at new values against OpenMM 8.6.1 and NumPy,
                and central differences of the energy in a tunable constant
                against the derivative of `observe` (D189) of `mdir run`
"""
import pathlib
import subprocess
import sys

import numpy as np
import mdir

root = sys.argv[1]
scenario = sys.argv[2]
KJ = 4.184


def expect(error, call, text=""):
    try:
        call()
    except error as exc:
        assert text in str(exc), str(exc)
        return str(exc)
    raise AssertionError(f"expected {error.__name__} ({text})")


def soft_term():
    # A repulsion between all pairs, in nm and kJ/mol, whose constants are
    # tunable.
    term = mdir.PairTerm()
    term.name, term.expression = "soft", "a*exp(-r/l)"
    term.constants = [("a", 2.0), ("l", 0.05)]
    return term


def spring_term():
    # Springs between the first heavy atoms of the dipeptide, a parameter per
    # tuple.
    term = mdir.TupleTerm()
    term.name, term.expression, term.arity = "spring", "0.5*k*(r - r0)^2", 2
    term.particles = np.array([[1, 4], [4, 6], [6, 8]], dtype=np.int64)
    term.parameters = [("k", np.array([500.0, 600.0, 600.0])),
                       ("r0", np.array([0.25, 0.26, 0.27]))]
    return term


def model(tunables=None, pme=True, terms=True):
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    # The defaults of before D[python-defaults], with which this was written.
    system.truncation = mdir.Truncation.Switch
    system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.7
    if pme:
        system.electrostatics = mdir.Electrostatics.PME
    if terms:
        system.pair_terms = [soft_term()]
        system.tuple_terms = [spring_term()]
    if tunables is not None:
        system.tunables = tunables(system)
    return system, state


def declarations(system, values=None):
    """Tied charges (one entry per atom name of the waters, each atom of the
    peptide its own), σ and ε of every type, a constant of the pair term,
    and the force constants of the springs, tied in two."""
    top = system.topology
    names = top.atom_names
    residues = [top.residue_names[r] for r in top.residue_indices]
    entries, charge_map = {}, []
    for i, (atom, residue) in enumerate(zip(names, residues)):
        key = (residue, atom) if residue == "WAT" else (residue, i)
        charge_map.append(entries.setdefault(key, len(entries)))
    charge_map = np.array(charge_map, dtype=np.int64)
    values = values or {}
    return [mdir.Tunable("q", "charge", map=charge_map, values=values.get("q")),
            mdir.Tunable("sigma", "sigma", values=values.get("sigma")),
            mdir.Tunable("epsilon", "epsilon", values=values.get("epsilon")),
            mdir.Tunable("soft_a", "a", term="soft", values=values.get("soft_a")),
            mdir.Tunable("k", "k", term="spring", map=np.array([0, 1, 1]),
                         values=values.get("k"))]


def compile_(system, state, target="CPU", precision="Double", leapfrog=False, npt=False,
             deterministic=True):
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    ensemble.com_period = 0  # the default of before D[python-defaults]
    integrator.timestep = 0.0005
    if leapfrog:
        integrator.method = mdir.IntegratorMethod.Leapfrog
    if npt:
        ensemble.kind, ensemble.temperature, ensemble.coupling_period = mdir.EnsembleKind.NPT, 300.0, 10
    execution.target, execution.precision = getattr(mdir.Target, target), getattr(mdir.Precision, precision)
    execution.deterministic = deterministic
    return mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())


def simulation(program):
    """A simulation whose runs are one part each, whatever a step takes: a
    part begins by ordering the particles anew (D196), so runs compared to
    the bit must be cut at the same steps."""
    sim = mdir.Simulation(program)
    sim.part_seconds = 1e9
    return sim


def new_values(sim):
    """Values that differ from the model's in every tunable."""
    v = dict(sim.tunables)
    rng = np.random.default_rng(7)
    q = v["q"] * (1.0 + 0.05 * rng.standard_normal(v["q"].shape))
    return {"q": q, "sigma": v["sigma"] * 1.02, "epsilon": v["epsilon"] * 0.9,
            "soft_a": v["soft_a"] * 3.0, "k": v["k"] * 1.5}


def state_of(system, snapshot):
    state = mdir.InitialState()
    state.positions, state.velocities, state.cell = snapshot.positions, snapshot.velocities, snapshot.cell
    return state


def difference(a, b):
    return [float(np.abs(getattr(a, f) - getattr(b, f)).max()) for f in ("positions", "velocities", "forces")]


def gromacs_model(tunables):
    """Propane and water with an NBFIX of CT and OW in [ nonbond_params ]
    (Inputs/gromacs), the Lennard-Jones cut without a shift or correction,
    a Coulomb cutoff."""
    gromacs = root + "/../gromacs"
    loaded = mdir.load_gromacs(gromacs + "/system.top", gromacs + "/system.gro", defines=["FLEXIBLE"])
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.8
    system.truncation = mdir.Truncation.None_
    system.dispersion = mdir.DispersionCorrection.None_
    system.tunables = tunables
    # A state without velocities began at rest before D[python-defaults];
    # the update of sigma and epsilon of this test is taken from rest.
    state.velocities = np.zeros((system.particle_count, 3))
    return system, state


def gromacs_excluded(n):
    """The pairs of particles of the GROMACS model that its topology
    excludes: up to three bonds apart in propane (1-4 pairs included, which
    have parameters of their own), and within a water."""
    top = open(root + "/../gromacs/system.top").read()
    bonds, counts = [], {}
    section = None
    for line in top.splitlines():
        line = line.split(";")[0].strip()
        if line.startswith("["):
            section = line.strip("[] ")
            continue
        if section == "bonds" and line:
            bonds.append(tuple(int(x) - 1 for x in line.split()[:2]))
        if section == "molecules" and line:
            counts[line.split()[0]] = int(line.split()[1])
    n_pro = 11
    graph = {i: set() for i in range(n_pro)}
    for i, j in bonds:
        graph[i].add(j); graph[j].add(i)
    excluded = set()
    for m in range(counts["PRO"]):
        o = n_pro * m
        for i in range(n_pro):
            seen, frontier = {i}, {i}
            for _ in range(3):
                frontier = {k for f in frontier for k in graph[f]} - seen
                seen |= frontier
            excluded |= {(o + min(i, j), o + max(i, j)) for j in seen if j != i}
    for w in range(counts["SOL"]):
        o = n_pro * counts["PRO"] + 3 * w
        excluded |= {(o, o + 1), (o, o + 2), (o + 1, o + 2)}
    assert n_pro * counts["PRO"] + 3 * counts["SOL"] == n
    return excluded


def run_nbfix(target, precision):
    """Per-type sigma and epsilon on a table with an NBFIX: an update equals
    a compile with the rule's values for the other pairs and the NBFIX's
    own, to the bit; the change of the energy at fixed positions against a
    NumPy sum of the Lennard-Jones that keeps the NBFIX."""
    import warnings
    T = mdir.Tunable

    def declarations(values=None):
        values = values or {}
        return [T("sigma", "sigma", mixing="geometric", values=values.get("sigma")),
                T("epsilon", "epsilon", values=values.get("epsilon"))]

    system, state = gromacs_model(declarations())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = simulation(compile_(system, state, target, precision))
    sim.run(6)
    sim.run(0, energy=True)
    at = sim.state()
    theta0 = dict(sim.tunables)
    top = system.topology
    names = top.type_names
    theta1 = {"sigma": theta0["sigma"] * np.array([1.05 if n in ("CT", "OW") else 0.98 for n in names]),
              "epsilon": theta0["epsilon"] * np.array([1.3 if n in ("CT", "OW") else 0.9 for n in names])}
    sim.tunables.update(theta1)
    updated = sim.state()
    du = updated.energies["potential"] - at.energies["potential"]
    sim.run(10, energy=True)
    after = sim.state()

    other, _ = gromacs_model(declarations(theta1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fresh = simulation(compile_(other, state_of(other, at), target, precision))
    fresh.run(0, energy=True)
    assert np.array_equal(fresh.state().forces, updated.forces)
    assert fresh.state().energies == updated.energies
    fresh.run(10, energy=True)
    d = difference(after, fresh.state())
    assert d == [0.0, 0.0, 0.0] and after.energies == fresh.state().energies, d
    print(f"{target} {precision}: NBFIX kept: an update at step 6 and 10 steps equal a compile with "
          f"the new values from that state to the bit")

    if target != "CPU" or precision != "Double":
        return
    # NumPy: the Lennard-Jones of the pairs not excluded within the cutoff,
    # σ and ε of the geometric rule but for CT-OW, the NBFIX's.
    excluded = gromacs_excluded(len(at.positions))
    types = top.particle_types
    x, box = at.positions, np.diag(at.cell.vectors)
    ct, ow = names.index("CT"), names.index("OW")
    c6, c12 = 2.60000e-03, 3.10000e-06
    fixed = ((c12 / c6) ** (1 / 6), c6 * c6 / (4 * c12))

    def energy(theta, keep=True):
        sig, eps = theta["sigma"], theta["epsilon"]
        total = 0.0
        for i in range(len(x) - 1):
            d = x[i + 1:] - x[i]
            d -= box * np.round(d / box)
            r = np.sqrt((d * d).sum(1))
            js = np.arange(i + 1, len(x))
            a, b = types[i], types[js]
            s_ij, e_ij = np.sqrt(sig[a] * sig[b]), np.sqrt(eps[a] * eps[b])
            if keep:
                pair = ((a == ct) & (b == ow)) | ((a == ow) & (b == ct))
                s_ij = np.where(pair, fixed[0], s_ij)
                e_ij = np.where(pair, fixed[1], e_ij)
            ok = (r < 0.8) & np.array([(i, j) not in excluded for j in js])
            sr6 = (s_ij[ok] / r[ok]) ** 6
            total += (4 * e_ij[ok] * (sr6 * sr6 - sr6)).sum()
        return total

    reference = energy(theta1) - energy(theta0)
    remixed = energy(theta1, False) - energy(theta0, False)
    print(f"NBFIX: the change of the energy at fixed positions {du:.9f}, NumPy with CT-OW kept "
          f"{reference:.9f} kJ/mol, difference {du - reference:.1e} (with CT-OW mixed by the rule "
          f"it would be {remixed:.6f})")
    assert abs(du - reference) < 1e-9 * max(1.0, abs(reference))
    assert abs(remixed - reference) > 1e-3


def lj_table_energy(x, box, types, sig, eps, excluded, cutoff=0.8):
    """The Lennard-Jones of the pairs not excluded within the cutoff, σ and
    ε from the (T, T) tables of the pairs of types."""
    total = 0.0
    for i in range(len(x) - 1):
        d = x[i + 1:] - x[i]
        d -= box * np.round(d / box)
        r = np.sqrt((d * d).sum(1))
        js = np.arange(i + 1, len(x))
        s_ij, e_ij = sig[types[i], types[js]], eps[types[i], types[js]]
        ok = (r < cutoff) & np.array([(i, j) not in excluded for j in js])
        sr6 = (s_ij[ok] / r[ok]) ** 6
        total += (4 * e_ij[ok] * (sr6 * sr6 - sr6)).sum()
    return total


def table_of(pairs, values, types):
    """The (T, T) table of the values of the sites `pairs` (Topology.type_pairs)."""
    table = np.zeros((types, types))
    table[pairs[:, 0], pairs[:, 1]] = values
    table[pairs[:, 1], pairs[:, 0]] = values
    return table


def run_pairs(target, precision):
    """The Lennard-Jones table tunable by pairs of types (#160): an update
    that breaks the combining rule (and gives a pair without Lennard-Jones
    some) equals a compile with the new values, to the bit; on the CPU in
    double precision, the change of the energy against a NumPy sum over the
    table, pair and per-type tunables declared together, and the correction
    for the dispersion against its formula."""
    import warnings
    T = mdir.Tunable

    def declarations(values=None):
        values = values or {}
        return [T("sig", "sigma_pair", values=values.get("sig")),
                T("eps", "epsilon_pair", values=values.get("eps"))]

    system, state = gromacs_model(declarations())
    top = system.topology
    names, pairs = top.type_names, top.type_pairs
    nt = len(names)
    # The sites: the flat upper triangle, (a, b) with a <= b.
    assert pairs.shape == (nt * (nt + 1) // 2, 2) and pairs.dtype == np.int64
    assert [tuple(p) for p in pairs] == [(a, b) for a in range(nt) for b in range(a, nt)]
    site = {(names[a], names[b]): k for k, (a, b) in enumerate(pairs)}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning: no pair keeps a value of its own
        sim = simulation(compile_(system, state, target, precision))
    plan = sim.program.plan["tunables"]
    assert [(d["parameter"], d["sites"], d["unit"]) for d in plan] == \
        [("sigma_pair", len(pairs), "nm"), ("epsilon_pair", len(pairs), "kJ/mol")]
    sim.run(6)
    sim.run(0, energy=True)
    at = sim.state()
    theta0 = dict(sim.tunables)
    # Every σ by 0.97 and ε by 1.1; CT-HC's ε doubled, off the rule; OW-HW,
    # without Lennard-Jones in the model, given some.
    sig1, eps1 = theta0["sig"] * 0.97, theta0["eps"] * 1.1
    eps1[site["CT", "HC"]] *= 2.0
    assert theta0["eps"][site["OW", "HW"]] == 0.0
    sig1[site["OW", "HW"]], eps1[site["OW", "HW"]] = 0.2, 0.05
    theta1 = {"sig": sig1, "eps": eps1}
    sim.tunables.update(theta1)
    updated = sim.state()
    du = updated.energies["potential"] - at.energies["potential"]
    sim.run(10, energy=True)
    after = sim.state()

    other, _ = gromacs_model(declarations(theta1))
    fresh = simulation(compile_(other, state_of(other, at), target, precision))
    fresh.run(0, energy=True)
    assert np.array_equal(fresh.state().forces, updated.forces)
    assert fresh.state().energies == updated.energies
    fresh.run(10, energy=True)
    d = difference(after, fresh.state())
    assert d == [0.0, 0.0, 0.0] and after.energies == fresh.state().energies, d
    print(f"{target} {precision}: pairs: an update at step 6 off the combining rule and 10 steps "
          f"equal a compile with the new values from that state to the bit")

    if target != "CPU" or precision != "Double":
        return
    types = top.particle_types
    x, box = at.positions, np.diag(at.cell.vectors)
    excluded = gromacs_excluded(len(x))

    def energy(theta):
        return lj_table_energy(x, box, types, table_of(pairs, theta["sig"], nt),
                               table_of(pairs, theta["eps"], nt), excluded)

    reference = energy(theta1) - energy(theta0)
    # The same change with CT-HC by the rule of the new values of CT-CT and
    # HC-HC.
    rule = dict(theta1, eps=theta1["eps"].copy())
    ct, hc = names.index("CT"), names.index("HC")
    rule["eps"][site["CT", "HC"]] = np.sqrt(theta1["eps"][site["CT", "CT"]] *
                                            theta1["eps"][site["HC", "HC"]])
    remixed = energy(rule) - energy(theta0)
    # The 1-4 pairs of propane (CT-CT, CT-HC, HC-HC of [ pairtypes ]) keep
    # their own parameters: the sum leaves them out with the excluded pairs.
    print(f"pairs: the change of the energy at fixed positions {du:.9f}, NumPy over the table "
          f"{reference:.9f} kJ/mol, difference {du - reference:.1e} (with CT-HC by the rule it "
          f"would be {remixed:.6f})")
    assert abs(du - reference) < 1e-9 * max(1.0, abs(reference))
    assert abs(remixed - reference) > 1e-3

    # Pair and per-type tunables together: the rule from the per-type values,
    # then the pairs that the pair tunables take (CT-OW, the NBFIX, and
    # CT-HC) have theirs; no pair keeps a value of its own, so no warning.
    take = np.full(len(pairs), -1, dtype=np.int64)
    take[site["CT", "OW"]], take[site["CT", "HC"]] = 0, 1

    def together(values=None):
        values = values or {}
        return [T("sigma", "sigma", mixing="geometric", values=values.get("sigma")),
                T("epsilon", "epsilon", values=values.get("epsilon")),
                T("sig", "sigma_pair", map=take, values=values.get("sig")),
                T("eps", "epsilon_pair", map=take, values=values.get("eps"))]

    both, _ = gromacs_model(together())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sim = simulation(compile_(both, state_of(both, at)))
    sim.run(0, energy=True)
    u0 = sim.state().energies["potential"]
    t0 = dict(sim.tunables)
    t1 = {"sigma": t0["sigma"] * 1.03, "epsilon": t0["epsilon"] * 0.8,
          "sig": t0["sig"] * np.array([0.95, 1.05]), "eps": t0["eps"] * np.array([1.4, 0.6])}
    sim.tunables.update(t1)
    u1 = sim.state().energies["potential"]

    def mixed(theta):
        sig = np.sqrt(np.outer(theta["sigma"], theta["sigma"]))
        eps = np.sqrt(np.outer(theta["epsilon"], theta["epsilon"]))
        for k, (a, b) in enumerate([(names.index("CT"), names.index("OW")), (ct, hc)]):
            sig[a, b] = sig[b, a] = theta["sig"][k]
            eps[a, b] = eps[b, a] = theta["eps"][k]
        return lj_table_energy(x, box, types, sig, eps, excluded)

    reference = mixed(t1) - mixed(t0)
    print(f"pairs with per-type tunables: the change of the energy {u1 - u0:.9f}, NumPy "
          f"{reference:.9f} kJ/mol, difference {u1 - u0 - reference:.1e}")
    assert abs(u1 - u0 - reference) < 1e-9 * max(1.0, abs(reference))

    # Refusals, and the warning of a pair that still keeps a value of its
    # own: CT-OW with per-type σ and a pair tunable of ε only.
    expect(mdir.InputError, lambda: T("x", "sigma_pair", mixing="geometric"), "for \"sigma\" only")
    short, _ = gromacs_model([T("x", "epsilon_pair", map=np.zeros(nt, dtype=np.int64))])
    expect(mdir.InputError, lambda: compile_(short, state_of(short, at)),
           f"the map has {nt} entries, and the parameter has {len(pairs)} sites")
    expect(mdir.InputError, lambda: sim.tunables.update({"eps": -t1["eps"]}), "at least 0")
    assert sim.tunables.version == 1
    partly, _ = gromacs_model(together()[:2] + [T("eps", "epsilon_pair", map=take)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compile_(partly, state_of(partly, at))
    messages = [str(w.message) for w in caught if "NBFIX" in str(w.message)]
    assert len(messages) == 1 and messages[0].endswith("keep them: CT-OW"), messages
    print("pairs: refusals and the warning of a pair that keeps a value of its own")

    # The correction for the dispersion at the new pair values against
    # ν (4π/V) Σ I_ij over the pairs not excluded, I = -C6 / (3 r_c³),
    # ν = N² / (N (N - 1) - 2 N_excluded) (D209): -(2π/3) N² <C6> / (V r_c³)
    # with the table's C6 of the pair that breaks the rule. The pair terms of
    # the Python model read r and their constants, not the table, so their
    # tails do not depend on these tunables.
    def corrected(theta):
        result = []
        for correction in (mdir.DispersionCorrection.None_, mdir.DispersionCorrection.EnergyPressure):
            other, _ = gromacs_model(declarations())
            other.dispersion = correction
            s = simulation(compile_(other, state_of(other, at)))
            s.tunables.update(theta)
            s.run(0, energy=True)
            result.append(s.state().energies["potential"])
        return result[1] - result[0]

    n = len(types)
    sig, eps = table_of(pairs, theta1["sig"], nt), table_of(pairs, theta1["eps"], nt)
    c6 = 4.0 * eps * sig ** 6
    counts = np.bincount(types, minlength=nt).astype(float)
    total = 0.5 * sum(counts[a] * (counts[b] - (a == b)) * c6[a, b]
                      for a in range(nt) for b in range(nt))
    total -= sum(c6[types[i], types[j]] for i, j in excluded)
    nu = n * n / (n * (n - 1.0) - 2.0 * len(excluded))
    volume, rc = np.prod(box), 0.8
    formula = nu * 4.0 * np.pi / volume * (-total / (3.0 * rc ** 3))
    mine = corrected(theta1)
    print(f"pairs: the correction for the dispersion at the new pair values {mine:.9f}, "
          f"the formula {formula:.9f} kJ/mol, relative difference {abs(mine / formula - 1):.1e}")
    assert abs(mine / formula - 1.0) < 1e-9


def run_cache(target, precision, work):
    """The compile cache (D212) and tunables: the values of tunables are
    data of the entry's buffers and arguments, not of the module whose
    bitcode keys a cached host object, so programs compiled with other
    values hit the same entry and run with their own values; an update
    compiles nothing and takes nothing from the cache."""
    import os
    # A directory of its own, whatever the suite gives the other tests.
    os.environ.pop("MDIR_COMPILE_CACHE", None)
    os.environ["MDIR_COMPILE_CACHE_DIR"] = str(work / "cache")
    system, state = model(declarations)
    first = simulation(compile_(system, state, target, precision))
    stats = first.compile_stats
    assert stats["cache_stored"] == 1 and stats["cache_hits"] == 0, stats
    values = new_values(first)
    first.run(4)
    first.tunables.update(values)
    assert first.compile_stats == stats, (first.compile_stats, stats)
    first.run(6, energy=True)

    # A compile with other values hits the entry of the first.
    other, _ = model(lambda s: declarations(s, values))
    cached = simulation(compile_(other, state, target, precision))
    assert cached.compile_stats["cache_hits"] == 1, cached.compile_stats
    # The same compile without the cache gives the same run to the bit.
    os.environ["MDIR_COMPILE_CACHE"] = "off"
    plain = simulation(compile_(other, state, target, precision))
    assert plain.compile_stats["cache_hits"] == 0 and plain.compile_stats["host_compiled"] == 1
    del os.environ["MDIR_COMPILE_CACHE"]
    for sim in (cached, plain):
        sim.run(10, energy=True)
    d = difference(cached.state(), plain.state())
    assert d == [0.0, 0.0, 0.0] and cached.state().energies == plain.state().energies, d
    # And its values are its own, not those of the first compile.
    assert all(np.array_equal(cached.tunables[k], values[k]) for k in values)
    reference = simulation(compile_(system, state, target, precision))
    reference.run(10, energy=True)
    assert reference.state().energies != cached.state().energies
    print(f"{target} {precision}: a compile with other values hits the cached object of the first, "
          f"and runs to the bit as one compiled without the cache; an update compiles nothing")


def run_declarations():
    system, state = model()
    n = system.particle_count
    top = system.topology
    assert top.charges.shape == (n,) and top.particle_types.shape == (n,)
    assert len(top.atom_names) == n and len(top.residue_names) == top.residue_count
    assert top.residue_indices.shape == (n,) and top.residue_indices[-1] > 0
    assert len(top.type_names) == int(top.particle_types.max()) + 1
    nt = len(top.type_names)
    assert top.type_pairs.shape == (nt * (nt + 1) // 2, 2)
    print(f"model arrays: {n} particles, {nt} types, "
          f"{int(top.residue_indices[-1]) + 1} residues")
    good = declarations(system)
    system.tunables = good
    assert [t.name for t in system.tunables] == ["q", "sigma", "epsilon", "soft_a", "k"]
    assert system.tunables[4].map.tolist() == [0, 1, 1] and system.tunables[1].mixing == "arithmetic"

    def refused(tunables, text, error=mdir.InputError):
        system.tunables = tunables
        return expect(error, lambda: compile_(system, state), text)

    T = mdir.Tunable
    cases = [
        ([T("1q", "charge")], "a tunable needs a name"),
        ([T("q", "charge"), T("q", "sigma")], "two tunables are named 'q'"),
        ([T("q", "mass")], "the parameter is \"charge\""),
        ([T("q", "charge"), T("p", "charge")], "is declared by another tunable"),
        ([T("x", "a", term="nothing")], "no pair or tuple term is named 'nothing'"),
        ([T("x", "b", term="soft")], "the pair term 'soft' has no constant 'b'"),
        ([T("x", "b", term="spring")], "the tuple term 'spring' has no parameter 'b'"),
        ([T("q", "charge", map=np.zeros(3, dtype=np.int64))], "the map has 3 entries"),
        ([T("q", "charge", map=np.full(n, -2))], "below -1"),
        ([T("q", "charge", map=np.full(n, -1))], "the map takes no site"),
        ([T("q", "charge", map=np.r_[np.zeros(n - 1, dtype=np.int64), 2])], "no site takes the entry 1"),
        ([T("q", "charge", map=np.zeros(n, dtype=np.int64))], "differ in the model"),
        ([T("q", "charge", values=np.zeros(2))], "`values` has 2 entries"),
        ([T("s", "sigma", values=-np.ones(nt))], "values of at least 0"),
    ]
    for tunables, text in cases:
        refused(tunables, text)
    expect(mdir.InputError, lambda: T("e", "epsilon", mixing="geometric"), "for \"sigma\" only")
    expect(mdir.InputError, lambda: T("s", "sigma", mixing="harmonic"), "arithmetic\" or \"geometric")
    expect(mdir.InputError, lambda: T("s", "sigma", map=np.zeros(3)), "int32 or int64")
    expect(mdir.InputError, lambda: T("s", "sigma", values=[0.3]), "lists are not accepted")
    print(f"declaration refusals: {len(cases) + 4} passed")
    # A table with an override (an NBFIX of [ nonbond_params ]): the pair
    # keeps its values, and compile warns once, naming it.
    import warnings
    other, other_state = gromacs_model([T("s", "sigma", mixing="geometric")])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compile_(other, other_state)
    messages = [str(w.message) for w in caught if "NBFIX" in str(w.message)]
    assert len(messages) == 1 and messages[0].endswith("keep them: CT-OW"), messages
    print(f"NBFIX: one warning: {messages[0]}")
    # Declaring is structural: the program goes stale.
    system.tunables = good
    program = compile_(system, state)
    assert not program.stale
    plan = program.plan["tunables"]
    assert [(d["name"], d["entries"], d["unit"]) for d in plan][:3] == [
        ("q", int(good[0].map.max()) + 1, "e"), ("sigma", nt, "nm"),
        ("epsilon", nt, "kJ/mol")], plan
    assert "md.lookup %t_tunable_constants" in program.ir
    system.tunables = good[:2]
    assert program.stale
    expect(mdir.StaleProgramError, program.check_current)
    # Without tunables the program reads no table of them.
    system.tunables = []
    assert "tunable_constants" not in compile_(system, state).ir
    print("declaring is structural: the program goes stale; the plan lists the tunables")


def run_updates(target, precision):
    system, state = model(declarations)
    sim = simulation(compile_(system, state, target, precision))
    assert sim.tunables.version == 0 and sim.tunables.history == [(0, 0)]
    values = new_values(sim)
    old = dict(sim.tunables)
    # An update during a run (velocity Verlet, NVE) against a simulation
    # compiled with the new values from the state at the update.
    sim.run(8)
    at_update = sim.state()
    version = sim.tunables.update(values)
    assert version == 1 and sim.tunables.history == [(0, 0), (8, 1)], sim.tunables.history
    refreshed = sim.state()
    assert refreshed.tunables_version == 1 and refreshed.energies is not None
    assert np.array_equal(refreshed.positions, at_update.positions)
    assert np.array_equal(refreshed.velocities, at_update.velocities)
    assert not np.array_equal(refreshed.forces, at_update.forces)
    sim.run(12, energy=True)
    updated = sim.state()

    other_system, _ = model(lambda s: declarations(s, values))
    # The compile with the new values evaluates the state first, without a
    # step (run(0, energy=True)), so that both runs cut their calls of the
    # entry at the same steps: a call that carries forces from the last
    # sums in another order than one that evaluates them (D196).
    fresh = simulation(compile_(other_system, state_of(other_system, at_update), target, precision))
    assert all(np.array_equal(fresh.tunables[k], values[k]) for k in values)
    assert fresh.run(0, energy=True) == 0 and fresh.step == 0
    start = fresh.state()
    assert np.array_equal(start.forces, refreshed.forces), np.abs(start.forces - refreshed.forces).max()
    assert start.energies == refreshed.energies, (start.energies, refreshed.energies)
    fresh.run(12, energy=True)
    recompiled = fresh.state()
    d = difference(updated, recompiled)
    assert d == [0.0, 0.0, 0.0] and updated.energies == recompiled.energies, (d, updated.energies,
                                                                             recompiled.energies)
    print(f"{target} {precision}: the forces and energies of an update at step 8 equal those of a "
          f"compile with the new values at that state to the bit (potential "
          f"{refreshed.energies['potential']:.6f} kJ/mol), and so do the state and the energies "
          f"12 steps later (potential {updated.energies['potential']:.6f} kJ/mol)")

    # Back to the old values: the energy of the state is that of the old
    # values, within the rounding of another order of the sums.
    sim2 = simulation(compile_(system, state, target, precision))
    sim2.run(4, energy=True)
    before = sim2.state().energies["potential"]
    sim2.tunables.update(values)
    sim2.tunables.update(old)
    after = sim2.state().energies["potential"]
    tolerance = 1e-9 if precision == "Double" else 1e-4
    assert abs(after - before) <= tolerance * abs(before), (before, after)
    print(f"{target} {precision}: there and back: the potential within {abs(after - before):.1e} "
          f"kJ/mol of {before:.6f}")

    # Against the model without tunables: the table of the combining rule
    # replaces Amber's within 2e-7, and the constants are read from a table.
    plain_system, plain_state = model()
    plain = simulation(compile_(plain_system, plain_state, target, precision))
    tunable = simulation(compile_(system, state, target, precision))
    plain.run(20, energy=True)
    tunable.run(20, energy=True)
    p, t = plain.state(), tunable.state()
    d = difference(p, t)
    rel = abs(p.energies["potential"] - t.energies["potential"]) / abs(p.energies["potential"])
    limit = 1e-7 if precision == "Double" else 1e-5
    assert rel < limit and d[0] < 1e-6, (rel, d)
    print(f"{target} {precision}: against the model without tunables after 20 steps: potential "
          f"within {rel:.1e} relative, positions within {d[0]:.1e} nm")

    # Refused updates change nothing.
    sim = tunable
    version, values_before = sim.tunables.version, dict(sim.tunables)
    T = len(values_before["sigma"])
    for change, text, error in [
            ({"nothing": np.zeros(1)}, "no tunable named 'nothing'", mdir.InputError),
            ({"sigma": np.ones(T + 1)}, "expected shape", mdir.InputError),
            ({"sigma": np.full(T, np.nan)}, "nonfinite", mdir.InputError),
            ({"epsilon": -np.ones(T)}, "values of at least 0", mdir.InputError),
            ({"sigma": np.ones(T), "q": np.full(values_before["q"].shape, 200.0)},
             "charges of less than 100 e", mdir.InputError),
            ({"sigma": [0.3] * T}, "lists are not accepted", mdir.InputError)]:
        expect(error, lambda: sim.tunables.update(change), text)
    assert sim.tunables.version == version
    assert all(np.array_equal(sim.tunables[k], values_before[k]) for k in values_before)
    expect(KeyError, lambda: sim.tunables["nothing"])
    plain_ = plain
    expect(mdir.InputError, lambda: plain_.tunables.update({"q": np.zeros(1)}), "declares no tunable")
    assert len(plain.tunables) == 0 and dict(plain.tunables) == {}
    print(f"{target} {precision}: 6 refused updates change neither values nor version")

    # The energy file takes the version as its last column; the frames take
    # theirs from the history.
    work = pathlib.Path(sys.argv[5] if len(sys.argv) > 5 else ".")
    path = work / f"versions-{target}-{precision}.dat"
    sim = simulation(compile_(system, state, target, precision))
    sim.reporters.append(mdir.EnergyReporter(str(path), 5))
    sim.run(10)
    sim.tunables["soft_a"] = np.array([5.0])
    sim.run(5)
    sim.tunables.update(soft_a=np.array([6.0]), k=values["k"])
    sim.run(5)
    sim.close_reporters()
    lines = path.read_text().splitlines()
    header = lines[0].lstrip("#").split()
    assert header[-1] == "tunables_version", header
    rows = [(int(l.split()[0]), int(l.split()[-1])) for l in lines if not l.startswith("#")]
    assert rows == [(0, 0), (5, 0), (10, 0), (15, 1), (20, 2)], rows
    assert sim.tunables.history == [(0, 0), (10, 1), (15, 2)], sim.tunables.history
    assert sim.state().tunables_version == 2
    print(f"{target} {precision}: versions {rows} in the energy file; history {sim.tunables.history}")
    # Without tunables the energy file has no such column.
    path = work / f"plain-{target}-{precision}.dat"
    plain.reporters.append(mdir.EnergyReporter(str(path), 5))
    plain.run(5)
    plain.close_reporters()
    assert "tunables_version" not in path.read_text().splitlines()[0]


def run_integrators(target, precision):
    # Leapfrog: the velocities are half a step behind the positions, and an
    # update keeps them; the forces are those of the new values.
    system, state = model(declarations)
    sim = simulation(compile_(system, state, target, precision, leapfrog=True))
    sim.run(6)
    before = sim.state()
    values = new_values(sim)
    old = dict(sim.tunables)
    sim.tunables.update(values)
    after = sim.state()
    assert np.array_equal(after.positions, before.positions)
    assert np.array_equal(after.velocities, before.velocities)
    assert after.velocity_offset == -0.5 and after.energies is None
    assert not np.array_equal(after.forces, before.forces)
    sim.tunables.update(old)
    back = sim.state()
    scale = float(np.abs(before.forces).max())
    tolerance = 1e-9 if precision == "Double" else 1e-3
    assert float(np.abs(back.forces - before.forces).max()) <= tolerance * scale
    sim.run(4)
    # Without tunables the half kick back of leapfrog is not a select: an
    # evaluation without a step after the first run is refused.
    plain_system, plain_state = model()
    plain = simulation(compile_(plain_system, plain_state, target, precision, leapfrog=True))
    plain.run(0, energy=True)
    assert plain.state().energies is not None and plain.state().velocity_offset == -0.5
    plain.run(2)
    expect(mdir.UnsupportedError, lambda: plain.run(0, energy=True), "leapfrog")
    print(f"{target} {precision}: leapfrog keeps its positions and velocities through an update; "
          f"the forces back at the old values within "
          f"{float(np.abs(back.forces - before.forces).max()):.1e} kJ/mol/nm")

    # NPT: an update before the first run against a compile with the new
    # values, to the bit: the barostat takes the virial and energy of the
    # constant terms of the new values (arguments of the entry).
    sim = simulation(compile_(system, state, target, precision, npt=True))
    values = new_values(sim)
    sim.tunables.update(values)
    assert sim.tunables.history == [(0, 0), (0, 1)]
    sim.run(30, energy=True)
    other_system, _ = model(lambda s: declarations(s, values))
    fresh = simulation(compile_(other_system, state, target, precision, npt=True))
    fresh.run(30, energy=True)
    a, b = sim.state(), fresh.state()
    d = difference(a, b)
    assert d == [0.0, 0.0, 0.0] and a.energies == b.energies, (d, a.energies, b.energies)
    assert "%baro_constant: f64" in compile_(system, state, target, precision, npt=True).ir
    print(f"{target} {precision}: NPT, an update before the first run equals a compile with the "
          f"new values to the bit after 30 steps (volume {a.energies['volume']:.6f} nm^3)")


def evaluated(sim):
    """The potential of the state, kJ/mol, after an evaluation or update."""
    energies = sim.state().energies
    assert energies is not None
    return energies["potential"]


def run_oracle(cli, work):
    from openmm import app, unit
    import openmm

    # MDIR: the dipeptide in water, PME on a fixed grid and β, the
    # Lennard-Jones cut without a shift and no correction, as OpenMM's
    # NonbondedForce takes them; charges, σ, and ε tunable.
    alpha, grid = 2.0, 72
    system, state = model(lambda s: [mdir.Tunable("q", "charge"), mdir.Tunable("sigma", "sigma"),
                                     mdir.Tunable("epsilon", "epsilon"),
                                     mdir.Tunable("soft_l", "l", term="soft"),
                                     mdir.Tunable("soft_a", "a", term="soft"),
                                     mdir.Tunable("k", "k", term="spring")])
    system.truncation = mdir.Truncation.None_
    system.dispersion = mdir.DispersionCorrection.None_
    system.pme_alpha, system.pme_grid = alpha, [grid] * 3
    sim = simulation(compile_(system, state))
    sim.run(0, energy=True)
    u0 = evaluated(sim)
    theta0 = dict(sim.tunables)
    rng = np.random.default_rng(11)
    theta1 = {"q": theta0["q"] * (1.0 + 0.1 * rng.standard_normal(theta0["q"].shape)),
              "sigma": theta0["sigma"] * (1.0 + 0.03 * rng.standard_normal(theta0["sigma"].shape)),
              "epsilon": theta0["epsilon"] * (1.0 + 0.2 * rng.random(theta0["epsilon"].shape)),
              "soft_a": np.array([7.0]), "soft_l": np.array([0.06]), "k": theta0["k"] * 2.0}
    # Two constants of one term, declared in the other order than the
    # term's: each is read from its own column of the table.
    assert sim.program.plan["tunables"][3]["name"] == "soft_l"
    sim.tunables.update(theta1)
    u1 = evaluated(sim)

    # OpenMM at the same positions, the new charges (1-4 products scaled by
    # 1/1.2 as the topology's), σ and ε of the types (Lorentz-Berthelot),
    # the 1-4 pairs' own σ and ε kept; the pair and tuple terms by NumPy.
    prmtop = app.AmberPrmtopFile(root + "/dipeptide.prmtop")
    types = system.topology.particle_types
    positions = state.positions
    cell = state.cell.vectors

    def openmm_energy(theta):
        omm = prmtop.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=0.8 * unit.nanometer,
                                  constraints=None, rigidWater=False)
        for force in omm.getForces():
            if isinstance(force, openmm.NonbondedForce):
                nb = force
        nb.setUseDispersionCorrection(False)
        nb.setUseSwitchingFunction(False)
        nb.setPMEParameters(alpha, grid, grid, grid)
        old = [nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
               for i in range(nb.getNumParticles())]
        for i in range(nb.getNumParticles()):
            nb.setParticleParameters(i, theta["q"][i], theta["sigma"][types[i]],
                                     theta["epsilon"][types[i]])
        for k in range(nb.getNumExceptions()):
            i, j, qq, sig, eps = nb.getExceptionParameters(k)
            qq = qq.value_in_unit(unit.elementary_charge ** 2)
            if qq != 0.0:
                scale = qq / (old[i] * old[j])
                nb.setExceptionParameters(k, i, j, scale * theta["q"][i] * theta["q"][j], sig, eps)
        omm.setDefaultPeriodicBoxVectors(*[openmm.Vec3(*row) for row in cell])
        context = openmm.Context(omm, openmm.VerletIntegrator(0.001),
                                 openmm.Platform.getPlatformByName("Reference"))
        context.setPositions(positions)
        return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)

    exclusions = set()
    omm = prmtop.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=0.8 * unit.nanometer)
    for force in omm.getForces():
        if isinstance(force, openmm.NonbondedForce):
            for k in range(force.getNumExceptions()):
                i, j = force.getExceptionParameters(k)[:2]
                exclusions.add((min(i, j), max(i, j)))
    box = np.diag(cell)

    def numpy_terms(theta):
        # The soft pair term over the pairs that are not excluded within the
        # cutoff, and the springs.
        x = positions
        total = 0.0
        for i in range(len(x) - 1):
            d = x[i + 1:] - x[i]
            d -= box * np.round(d / box)
            r = np.sqrt((d * d).sum(1))
            js = np.arange(i + 1, len(x))
            keep = (r < 0.8) & np.array([(i, j) not in exclusions for j in js])
            total += (theta["soft_a"][0] * np.exp(-r[keep] / theta["soft_l"][0])).sum()
        k = theta["k"][[0, 1, 1]]
        for (i, j), kk, r0 in zip([[1, 4], [4, 6], [6, 8]], k, [0.25, 0.26, 0.27]):
            d = x[j] - x[i]
            d -= box * np.round(d / box)
            total += 0.5 * kk * (np.sqrt((d * d).sum()) - r0) ** 2
        return total

    reference = [openmm_energy(t) + numpy_terms(t) for t in (theta0, theta1)]
    du, dref = u1 - u0, reference[1] - reference[0]
    print(f"oracle: U(theta0) {u0:.6f}, OpenMM + NumPy {reference[0]:.6f} kJ/mol, "
          f"difference {u0 - reference[0]:.2e}")
    print(f"oracle: U(theta1) {u1:.6f}, OpenMM + NumPy {reference[1]:.6f} kJ/mol, "
          f"difference {u1 - reference[1]:.2e}")
    print(f"oracle: U(theta1) - U(theta0) {du:.6f}, OpenMM + NumPy {dref:.6f} kJ/mol, "
          f"difference {du - dref:.2e}")
    # B-splines of order 4 (MDIR) against 5 (OpenMM): on a grid of 72 over
    # 2.7 nm with β = 2/nm the two sums of the model without tunables differ
    # by 4.6e-4 kJ/mol (by 0.03 with β = 3.5/nm, 0.14 on a grid of 48).
    assert abs(u0 - reference[0]) < 2e-3 and abs(u1 - reference[1]) < 2e-3
    assert abs(du - dref) < 1e-3

    # The correction for the dispersion at the new σ and ε against its
    # formula: -(2π/3) N² <C6> / (V r_c³), <C6> over the pairs that are not
    # excluded.
    def with_dispersion(theta):
        other, other_state = model(lambda s: [mdir.Tunable("sigma", "sigma", values=theta["sigma"]),
                                              mdir.Tunable("epsilon", "epsilon",
                                                           values=theta["epsilon"])],
                                    terms=False)
        other.truncation = mdir.Truncation.None_
        other.pme_alpha, other.pme_grid = alpha, [grid] * 3
        result = []
        for correction in (mdir.DispersionCorrection.None_, mdir.DispersionCorrection.EnergyPressure):
            other.dispersion = correction
            s = simulation(compile_(other, other_state))
            s.run(0, energy=True)
            result.append(evaluated(s))
        return result[1] - result[0]

    n = len(types)
    sig, eps = theta1["sigma"], theta1["epsilon"]

    def c6(a, b):
        return 4.0 * np.sqrt(eps[a] * eps[b]) * (0.5 * (sig[a] + sig[b])) ** 6

    counts = np.bincount(types, minlength=len(sig)).astype(float)
    total = sum(counts[a] * (counts[b] - (a == b)) * c6(a, b)
                for a in range(len(sig)) for b in range(len(sig)))
    total -= 2.0 * sum(c6(types[i], types[j]) for i, j in exclusions)
    mean = total / (n * (n - 1.0) - 2.0 * len(exclusions))
    formula = -2.0 * np.pi / (3.0 * np.prod(box)) * n * n * mean / 0.8 ** 3
    mine = with_dispersion(theta1)
    print(f"oracle: the correction for the dispersion at the new sigma and epsilon {mine:.9f}, "
          f"the formula {formula:.9f} kJ/mol, relative difference {abs(mine / formula - 1):.1e}")
    assert abs(mine / formula - 1.0) < 1e-9

    # Central differences of U in a tunable constant of a pair term, by two
    # updates at fixed positions, against the derivative of `observe` of
    # mdir run at the same positions (D189), with the tail and the estimate
    # of the shift of D209 and D210.
    system, state = model(lambda s: [mdir.Tunable("soft_l", "l", term="soft")])
    system.truncation = mdir.Truncation.Shift
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    sim = simulation(compile_(system, state))
    sim.run(0, energy=True)
    l, h = 0.05, 1e-5
    sim.tunables["soft_l"] = np.array([l + h])
    plus = evaluated(sim)
    sim.tunables["soft_l"] = np.array([l - h])
    minus = evaluated(sim)
    fd = (plus - minus) / (2.0 * h) / (KJ * 10.0)  # kcal/mol/Å
    control = work / "observe.toml"
    control.write_text(f"""[input]
topology    = "{root}/dipeptide.prmtop"
coordinates = "{root}/dipeptide.inpcrd"
[output]
energy_interval = 1
observables = "observe.obs"
energy = "observe.dat"
[energy]
cutoff            = 8.0
pairlist_distance = 9.0
electrostatics    = "PME"
lennard_jones_modifier = "POTENTIAL_SHIFT"
dispersion_correction = "ENERGY_PRESSURE"
[[energy.pair]]
name       = "soft"
expression = "a*exp(-r/l)"
observe    = ["l"]
a = {2.0 / KJ!r}
l = 0.5
[dynamics]
time_step = 0.0005
steps     = 1
[ensemble]
ensemble = "NVE"
[boundary]
type = "PERIODIC"
[execution]
target = "CPU"
precision = "DOUBLE"
""")
    subprocess.run([cli, "run", str(control)], cwd=work, check=True, stdout=subprocess.DEVNULL)
    rows = [line.split() for line in (work / "observe.obs").read_text().splitlines()
            if not line.startswith("#")]
    header = (work / "observe.obs").read_text().splitlines()[0].lstrip("#").split()
    observed = float(rows[0][header.index("soft.d_l")])
    print(f"derivative: central differences {fd:.6f}, observe of mdir run {observed:.6f} "
          f"kcal/mol/Å, relative difference {abs(fd / observed - 1):.1e}")
    assert abs(fd / observed - 1.0) < 1e-6


if scenario == "declarations":
    run_declarations()
elif scenario == "updates":
    run_updates(sys.argv[3], sys.argv[4])
elif scenario == "integrators":
    run_integrators(sys.argv[3], sys.argv[4])
elif scenario == "cache":
    run_cache(sys.argv[3], sys.argv[4], pathlib.Path(sys.argv[5]))
elif scenario == "nbfix":
    run_nbfix(sys.argv[3], sys.argv[4])
elif scenario == "pairs":
    run_pairs(sys.argv[3], sys.argv[4])
elif scenario == "oracle":
    run_oracle(sys.argv[3], pathlib.Path(sys.argv[4]))
else:
    raise SystemExit(f"unknown scenario {scenario}")
print(f"{scenario} passed")
