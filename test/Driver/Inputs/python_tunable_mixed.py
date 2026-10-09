"""The derivative of the energy in the charges together with the other
tunables of one program (D230, #256, docs/python-gradient.md).

Usage: python_tunable_mixed.py ROOT SCENARIO [TARGET PRECISION]

Scenarios:
  refusals  what the Python model cannot declare beside the charges, and with
            which error (CPU)
  test      the combinations of the test: the charges free with PME and tied
            by a map under a Coulomb cutoff, each with constants of a pair
            term and with every other tunable at once
  sweep     every combination: the charges free and tied, with PME and under
            a Coulomb cutoff, with each kind of the other tunables and with
            all of them

Each combination is one program with the charges and the other tunables,
evaluated at the state it begins at, where three programs hold the same
state to the bit. Checked:

- the derivative against central differences of the energy that it is the
  derivative of (`gradient().energy`), extrapolated from two steps
  (Richardson), at the three entries of each tunable that are largest in
  magnitude, relative to the largest entry of the tunable;
- the derivative in the charges against that of a program that declares
  the charges alone, and the derivative in the other tunables against that
  of a program that declares them alone: the derivative in one tunable does
  not depend on whether another is declared.
"""
import sys
import warnings

import numpy as np
import mdir

root = sys.argv[1]
scenario = sys.argv[2]

SPRINGS = np.array([[1, 4], [4, 6], [6, 8], [1, 6], [1, 8], [4, 8]], dtype=np.int64)
K_MAP = np.array([0, 1, 1, 2, -1, 0])

# The other tunables, by kind.
KINDS = ["constant", "pairs", "types", "tuples", "external", "observed", "all"]
LABELS = {
    "constant": "constants a, l of the pair term a exp(-r/l)",
    "pairs": "sigma and epsilon of three pairs of types",
    "types": "per-type sigma and epsilon",
    "tuples": "parameters k, r0 of springs over pairs",
    "external": "sigma of three pairs, and an external term e0 q z that reads the charges",
    "observed": "the constant l, observed, with an observed external term",
    "all": "all of these at once",
}


def expect(error, call, text=""):
    try:
        call()
    except error as exc:
        assert text in str(exc), str(exc)
        return str(exc)
    raise AssertionError(f"expected {error.__name__} ({text})")


def charge_map(system):
    """One entry per atom name of the waters, and one per atom of the rest."""
    top = system.topology
    residues = [top.residue_names[r] for r in top.residue_indices]
    entries, result = {}, []
    for i, (atom, residue) in enumerate(zip(top.atom_names, residues)):
        key = (residue, atom) if residue == "WAT" else (residue, i)
        result.append(entries.setdefault(key, len(entries)))
    return np.array(result, dtype=np.int64)


def type_epsilon(cache=[]):
    """The epsilon of each type of the model: the values of a tunable that
    takes them all."""
    if not cache:
        loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
        system, state = loaded.make_system(), loaded.make_state()
        system.dispersion = mdir.DispersionCorrection.None_
        system.tunables = [mdir.Tunable("epsilon", "epsilon")]
        program = mdir.compile(system, state, mdir.Integrator(), mdir.Ensemble(),
                               mdir.Execution(), mdir.Schedule())
        cache.append(mdir.Simulation(program).tunables["epsilon"].copy())
    return cache[0]


def model(pme, kind):
    """The dipeptide in water with the terms of `kind`, and its tunables
    other than the charges."""
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.7
    system.electrostatics = mdir.Electrostatics.PME if pme else mdir.Electrostatics.Cutoff
    system.dispersion = mdir.DispersionCorrection.None_
    top = system.topology
    others = []
    has = lambda name: kind in (name, "all")
    if has("constant") or has("observed"):
        soft = mdir.PairTerm()
        soft.name, soft.expression = "soft", "a*exp(-r/l)"
        soft.constants = [("a", 2.0), ("l", 0.05)]
        if has("observed"):
            soft.observe = ["l"]
        system.pair_terms = [soft]
        others.append(mdir.Tunable("soft_l", "l", term="soft"))
        if has("constant"):
            others.append(mdir.Tunable("soft_a", "a", term="soft"))
    if has("pairs") or has("external"):
        taken = np.full(len(top.type_pairs), -1)
        taken[0], taken[3], taken[len(top.type_names) + 2] = 1, 0, 2
        others.append(mdir.Tunable("sigma_pair", "sigma_pair", map=taken))
        if has("pairs"):
            others.append(mdir.Tunable("epsilon_pair", "epsilon_pair", map=taken))
    if has("types"):
        # The hydrogen of the water has no Lennard-Jones, and sqrt(x y) has
        # no derivative at x = 0: its type is left out.
        epsilon = type_epsilon()
        keep = np.where(epsilon > 0.0, np.cumsum(epsilon > 0.0) - 1, -1)
        others.append(mdir.Tunable("sigma", "sigma", map=keep))
        others.append(mdir.Tunable("epsilon", "epsilon", map=keep))
    if has("tuples"):
        spring = mdir.TupleTerm()
        spring.name, spring.expression, spring.arity = "spring", "0.5*k*(r - r0)^2", 2
        spring.particles = SPRINGS
        # The sites that an entry of the map ties have one value, so that
        # the program without the tunable has the same energy.
        spring.parameters = [("k", np.array([450.0, 600.0, 600.0, 310.0, 200.0, 450.0])),
                             ("r0", np.array([0.25, 0.26, 0.27, 0.3, 0.35, 0.4]))]
        system.tuple_terms = [spring]
        others.append(mdir.Tunable("k", "k", term="spring", map=K_MAP))
        others.append(mdir.Tunable("r0", "r0", term="spring"))
    externals = []
    if has("external"):
        # A uniform field along z: a sum over particles that reads the
        # charges, whose derivative in them is a map.
        field = mdir.ExternalTerm()
        field.name, field.expression, field.selection = "field", "e0*q*z", ":WAT"
        field.constants = [("e0", 25.0)]
        externals.append(field)
    if has("observed"):
        upper = mdir.ExternalTerm()
        upper.name, upper.expression, upper.selection = "upper", "0.5*k*max(0, z - z0)^2", ":WAT"
        upper.constants = [("k", 4184.0), ("z0", 2.0)]
        upper.observe = ["z0", "k"]
        externals.append(upper)
    system.external_terms = externals
    return system, state, others


def simulation(system, state, tunables, target, precision):
    system.tunables = tunables
    system.tunable_gradient = True
    integrator, execution = mdir.Integrator(), mdir.Execution()
    integrator.timestep = 0.0005
    execution.target = getattr(mdir.Target, target)
    execution.precision = getattr(mdir.Precision, precision)
    execution.deterministic = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        program = mdir.compile(system, state, integrator, mdir.Ensemble(), execution,
                               mdir.Schedule())
    assert all(d["gradient"] == "rule" for d in program.plan["tunables"]), program.plan
    sim = mdir.Simulation(program)
    sim.part_seconds = 1e9
    return sim


def differences(sim, g, name, step):
    """The largest difference between the derivative in the tunable `name`
    and central differences of the energy, extrapolated from the steps h
    and h / 2 with h = `step` times each value (Richardson), at the three
    entries that are largest in magnitude, relative to the largest; and the
    largest h."""
    v0 = sim.tunables[name].copy()
    scale = np.abs(g[name]).max()
    worst, largest = 0.0, 0.0
    for m in np.argsort(-np.abs(g[name]), kind="stable")[:3]:
        def central(h):
            e = []
            for sign in (1.0, -1.0):
                v = v0.copy()
                v[m] += sign * h
                sim.tunables[name] = v
                e.append(sim.tunables.gradient().energy)
            return (e[0] - e[1]) / (2.0 * h)
        h = step * max(abs(v0[m]), 1e-3)
        fd = (4.0 * central(0.5 * h) - central(h)) / 3.0
        worst = max(worst, abs(g[name][m] - fd) / scale)
        largest = max(largest, h)
    sim.tunables[name] = v0
    return worst, largest


def closeness(a, b):
    """0 for arrays equal to the bit, else their largest difference relative
    to the largest entry."""
    if np.array_equal(a, b):
        return 0.0
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def combination(pme, tied, kind, target, precision):
    double = precision == "Double"
    name = (f"{'PME' if pme else 'cutoff'}, charges {'tied (25 entries)' if tied else 'free'}, "
            f"{LABELS[kind]}")

    def charges(system):
        return mdir.Tunable("q", "charge", map=charge_map(system)) if tied \
            else mdir.Tunable("q", "charge")

    system, state, others = model(pme, kind)
    both = simulation(system, state, [charges(system)] + others, target, precision)
    g = both.tunables.gradient()
    assert g.step == 0 and not g.zero, g.zero
    assert g["q"].shape == (25 if tied else 1168,)

    # Against the programs that declare one of the two.
    system, state, _ = model(pme, kind)
    alone = simulation(system, state, [charges(system)], target, precision).tunables.gradient()
    same_q = closeness(g["q"], alone["q"])
    system, state, others = model(pme, kind)
    rest = simulation(system, state, others, target, precision).tunables.gradient()
    same_rest = max(closeness(g[t.name], rest[t.name]) for t in others)
    # The energy is that of the program of the other tunables. That of the
    # program of the charges alone is the same but where per-type sigma and
    # epsilon are tunable: their table then follows the combining rule,
    # which the eight digits of the file's table leave within 2e-7 (D213).
    same_energy = abs(g.energy - rest.energy) / abs(g.energy)
    if kind not in ("types", "all"):
        same_energy = max(same_energy, abs(g.energy - alone.energy) / abs(g.energy))

    # An observed constant that is a tunable: its column of `observe` is
    # the derivative (D232).
    if kind in ("observed", "all"):
        both.run(0, energy=True)
        column = both.state().observables["soft.d_l"]
        assert abs(column - g["soft_l"][0]) <= (1e-12 if double else 1e-5) * abs(column), \
            (column, g["soft_l"])

    # Against central differences of the energy.
    step_q = 1e-3 if double else 5e-2
    step_rest = 1e-3 if double else 2e-2
    fd_q, h_q = differences(both, g, "q", step_q)
    fd_rest = 0.0
    for tunable in others:
        fd_rest = max(fd_rest, differences(both, g, tunable.name, step_rest)[0])
    again = both.tunables.gradient()
    assert np.array_equal(again["q"], g["q"]), "the derivative after the differences"

    print(f"{target} {precision}: {name}: central differences q {fd_q:.1e} "
          f"(h {step_q:g} of the value, at most {h_q:.1e} e), others {fd_rest:.1e} "
          f"(h {step_rest:g} of the value); against the programs of one tunable q "
          f"{same_q:.1e}, others {same_rest:.1e}, energy {same_energy:.1e}")
    tolerance_fd = 1e-7 if double else 5e-4
    assert fd_q < tolerance_fd and fd_rest < tolerance_fd, (fd_q, fd_rest)
    # The derivatives of the programs of one tunable, to the bit: the
    # programs are compiled in the deterministic mode, and each derivative
    # has the kernels and the sums that it has alone. The energy adds its
    # terms as each program forms them (the 1-4 pairs from the charges of
    # their members where the charges are a tunable, D230), to rounding.
    assert same_q == 0.0 and same_rest == 0.0, (same_q, same_rest)
    assert same_energy < (1e-13 if double else 1e-6), same_energy
    return fd_q, fd_rest, same_q, same_rest


def run_refusals():
    # A constant of an external term is not a tunable: `term` names a pair
    # term or a term over tuples.
    system, state, others = model(True, "external")
    system.tunables = [mdir.Tunable("q", "charge"), mdir.Tunable("e0", "e0", term="field")]
    system.tunable_gradient = True
    integrator, execution = mdir.Integrator(), mdir.Execution()
    compile_ = lambda: mdir.compile(system, state, integrator, mdir.Ensemble(), execution,
                                    mdir.Schedule())
    print("refusals: a constant of an external term:",
          expect(mdir.InputError, compile_, "no pair or tuple term is named 'field'"))
    # A pair term of the Python model reads r and its constants, not the
    # charges of its particles.
    system, state, others = model(True, "constant")
    term = system.pair_terms[0]
    term.expression = "a*q1*q2*exp(-r/l)"
    system.pair_terms = [term]
    system.tunables = [mdir.Tunable("q", "charge")] + others
    system.tunable_gradient = True
    print("refusals: a pair term that reads the charges:",
          expect(mdir.InputError, compile_, "undeclared parameter 'q1'"))
    # The Python model has no reaction field.
    assert sorted(mdir.Electrostatics.__members__) == ["Cutoff", "PME"]
    print("refusals passed")


if scenario == "refusals":
    run_refusals()
else:
    target, precision = sys.argv[3], sys.argv[4]
    if scenario == "test":
        cases = [(True, False, "constant"), (False, True, "constant"),
                 (True, False, "all"), (False, True, "all")]
    else:
        cases = [(pme, tied, kind) for kind in KINDS for pme in (True, False)
                 for tied in (False, True)]
    for pme, tied, kind in cases:
        combination(pme, tied, kind, target, precision)
    print(f"{target} {precision}: {len(cases)} combinations")
    print(f"{scenario} passed")
