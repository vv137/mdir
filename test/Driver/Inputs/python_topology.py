"""Read-only topology views and mask selection in the Python model
(D[python-topology]): the values of the views against parsers of the files
written here, `select` against the selections that the control file's path
makes (`mdir-model-test --selections`), the constraints after preparation,
and copies, dtypes, and snapshots.

Usage: python_topology.py INPUTS CHARMM WORK MDIR-MODEL-TEST, with INPUTS
test/Driver/Inputs, CHARMM the files of charmm.test, and WORK a directory of
its own."""
import pathlib
import re
import subprocess
import sys
from collections import Counter

import numpy as np
import mdir

inputs = pathlib.Path(sys.argv[1])
charmm = pathlib.Path(sys.argv[2])
work = pathlib.Path(sys.argv[3])
model_test = sys.argv[4]
work.mkdir(parents=True, exist_ok=True)


def expect(error, call, text=""):
    try:
        call()
    except error as exc:
        assert text in str(exc), str(exc)
        return str(exc)
    raise AssertionError(f"expected {error.__name__}")


def rows(array):
    """A multiset of the rows of an integer array."""
    return Counter(map(tuple, np.asarray(array).tolist()))


# --- Oracles: parsers of the files ---------------------------------------

def read_prmtop(path):
    """The sections of a prmtop by their fixed-width %FORMAT."""
    sections, name, width, kind, data = {}, None, 0, "", []
    for line in pathlib.Path(path).read_text().splitlines():
        if line.startswith("%FLAG"):
            if name:
                sections[name] = data
            name, data = line.split()[1], []
        elif line.startswith("%FORMAT"):
            fmt = re.search(r"\((\d+)([aIE])(\d+)", line)
            kind, width = fmt.group(2), int(fmt.group(3))
        elif name and not line.startswith("%"):
            for k in range(0, len(line), width):
                field = line[k:k + width]
                if not field.strip():
                    continue
                data.append(field.strip() if kind == "a" else
                            int(field) if kind == "I" else float(field))
    sections[name] = data
    return sections


def amber_oracle(path):
    p = read_prmtop(path)
    n = len(p["ATOM_NAME"])
    starts = [s - 1 for s in p["RESIDUE_POINTER"]]
    residue = np.searchsorted(starts, np.arange(n), side="right") - 1
    bonds = p["BONDS_INC_HYDROGEN"] + p["BONDS_WITHOUT_HYDROGEN"]
    angles = p["ANGLES_INC_HYDROGEN"] + p["ANGLES_WITHOUT_HYDROGEN"]
    dihedrals = p["DIHEDRALS_INC_HYDROGEN"] + p["DIHEDRALS_WITHOUT_HYDROGEN"]
    hydrogen = [(bonds[k] // 3, bonds[k + 1] // 3)
                for k in range(0, len(p["BONDS_INC_HYDROGEN"]), 3)]
    return dict(
        names=p["ATOM_NAME"], masses=np.array(p["MASS"]),
        charges=np.array(p["CHARGE"]) / 18.2223,
        atomic_numbers=np.array(p["ATOMIC_NUMBER"]),
        types=np.array(p["ATOM_TYPE_INDEX"]) - 1,
        residue_names=p["RESIDUE_LABEL"], residue_starts=np.array(starts),
        residue_indices=residue,
        bonds=rows([(bonds[k] // 3, bonds[k + 1] // 3) for k in range(0, len(bonds), 3)]),
        angles=rows([(angles[k] // 3, angles[k + 1] // 3, angles[k + 2] // 3)
                     for k in range(0, len(angles), 4)]),
        dihedrals=rows([(dihedrals[k] // 3, dihedrals[k + 1] // 3,
                         abs(dihedrals[k + 2]) // 3, abs(dihedrals[k + 3]) // 3)
                        for k in range(0, len(dihedrals), 5)]),
        impropers=sum(dihedrals[k + 3] < 0 for k in range(0, len(dihedrals), 5)),
        hydrogen_bonds=hydrogen)


def gromacs_oracle(top, defines):
    """[ atoms ], [ bonds ], [ angles ], [ dihedrals ], and [ settles ] of
    each molecule type, times [ molecules ], with #ifdef/#else/#endif and
    #include of files next to `top`."""
    types, molecules, active, section, current = {}, [], [True], None, None

    def lines(path):
        for raw in path.read_text().replace("\\\n", " ").splitlines():
            line = raw.split(";")[0].strip()
            if line.startswith("#include"):
                yield from lines(path.parent / line.split('"')[1])
            elif line:
                yield line
    for line in lines(pathlib.Path(top)):
        if line.startswith("#ifdef"):
            active.append(active[-1] and line.split()[1] in defines)
        elif line.startswith("#else"):
            active[-1] = not active[-1] and active[-2]
        elif line.startswith("#endif"):
            active.pop()
        elif not active[-1] or line.startswith("#"):
            continue
        elif line.startswith("["):
            section = line.strip("[] ")
        elif section == "moleculetype":
            current = types[line.split()[0]] = {"atoms": [], "bonds": [], "angles": [],
                                                  "dihedrals": [], "settles": []}
        elif section in ("atoms", "bonds", "angles", "dihedrals", "settles") and current:
            current[section].append(line.split())
        elif section == "molecules":
            name, count = line.split()
            molecules.append((name, int(count)))
    result = {"names": [], "charges": [], "residue_names": [], "residue_starts": [],
              "bonds": Counter(), "angles": Counter(), "quads": set(), "settles": []}
    offset = 0
    for name, count in molecules:
        mol = types[name]
        for _ in range(count):
            last = None
            for k, atom in enumerate(mol["atoms"]):
                result["names"].append(atom[4])
                result["charges"].append(float(atom[6]))
                if atom[2] != last:
                    result["residue_names"].append(atom[3])
                    result["residue_starts"].append(offset + k)
                    last = atom[2]
            for section, arity in (("bonds", 2), ("angles", 3)):
                for row in mol[section]:
                    result[section][tuple(offset + int(x) - 1 for x in row[:arity])] += 1
            for row in mol["dihedrals"]:
                result["quads"].add(tuple(offset + int(x) - 1 for x in row[:4]))
            for row in mol["settles"]:
                o = offset + int(row[0]) - 1
                result["settles"].append((o, o + 1, o + 2))
            offset += len(mol["atoms"])
    return result


def psf_oracle(path):
    """The sections of a PSF of the XPLOR kind."""
    text = pathlib.Path(path).read_text().splitlines()
    sections, k = {}, 0
    while k < len(text):
        m = re.match(r"\s*(\d+)(?:\s+\d+)?\s+!(\w+)", text[k])
        if not m:
            k += 1
            continue
        count, name, k = int(m.group(1)), m.group(2), k + 1
        if name == "NTITLE":
            k += count
            continue
        if name == "NATOM":
            sections[name] = [text[k + i].split() for i in range(count)]
            k += count
            continue
        numbers = []
        while k < len(text) and text[k].strip():
            numbers += [int(x) for x in text[k].split()]
            k += 1
        sections[name] = numbers
    atoms = sections["NATOM"]
    group = lambda name, arity: [tuple(x - 1 for x in sections[name][i:i + arity])
                                 for i in range(0, len(sections[name]), arity)]
    starts, names, last = [], [], None
    for i, a in enumerate(atoms):
        if (a[1], a[2]) != last:
            starts.append(i)
            names.append(a[3])
            last = (a[1], a[2])
    return dict(names=[a[4] for a in atoms], types=[a[5] for a in atoms],
                charges=np.array([float(a[6]) for a in atoms]),
                masses=np.array([float(a[7]) for a in atoms]),
                residue_names=names, residue_starts=np.array(starts),
                bonds=rows(group("NBOND", 2)), angles=rows(group("NTHETA", 3)),
                quads=set(group("NPHI", 4)), impropers=set(group("NIMPHI", 4)))


# --- Values of the views ---------------------------------------------------

def check_common(top, n):
    """Shapes, dtypes, and read-only flags of a view of `n` particles."""
    arrays = {"atomic_numbers": (np.int64, (n,)), "masses": (np.float64, (n,)),
              "charges": (np.float64, (n,)), "particle_types": (np.int64, (n,)),
              "residue_indices": (np.int64, (n,)),
              "residue_starts": (np.int64, (top.residue_count,)),
              "bonds": (np.int64, (None, 2)), "angles": (np.int64, (None, 3)),
              "dihedrals": (np.int64, (None, 4)), "improper_dihedrals": (np.bool_, (None,)),
              "harmonic_impropers": (np.int64, (None, 4)),
              "virtual_sites": (np.int64, (None, 4))}
    for name, (dtype, shape) in arrays.items():
        a, b = getattr(top, name), getattr(top, name)
        assert a.dtype == dtype, (name, a.dtype)
        assert len(a.shape) == len(shape) and all(s is None or s == t for s, t in zip(shape, a.shape)), (name, a.shape)
        assert not a.flags.writeable and a.flags.c_contiguous, name
        assert a is not b and not np.shares_memory(a, b), name
        expect(ValueError, lambda: a.__setitem__(..., 0))
    assert top.improper_dihedrals.shape[0] == top.dihedrals.shape[0]
    for name in ("atom_names", "type_names", "residue_names"):
        names = getattr(top, name)
        assert isinstance(names, list) and all(isinstance(s, str) for s in names), name
    assert len(top.atom_names) == n and len(top.residue_names) == top.residue_count
    assert len(top.type_names) == int(top.particle_types.max()) + 1
    assert top.particle_count == n


def amber():
    o = amber_oracle(inputs / "dipeptide/dipeptide.prmtop")
    loaded = mdir.load_amber(str(inputs / "dipeptide/dipeptide.prmtop"),
                             str(inputs / "dipeptide/dipeptide.inpcrd"))
    top = loaded.topology
    n = len(o["names"])
    check_common(top, n)
    assert top.atom_names == o["names"] and top.residue_names == o["residue_names"]
    assert np.array_equal(top.masses, o["masses"])
    assert np.array_equal(top.charges, o["charges"])
    assert np.array_equal(top.atomic_numbers, o["atomic_numbers"])
    assert np.array_equal(top.particle_types, o["types"])
    assert np.array_equal(top.residue_starts, o["residue_starts"])
    assert np.array_equal(top.residue_indices, o["residue_indices"])
    assert rows(top.bonds) == o["bonds"], "bonds"
    assert rows(top.angles) == o["angles"], "angles"
    assert rows(top.dihedrals) == o["dihedrals"], "dihedrals"
    assert int(top.improper_dihedrals.sum()) == o["impropers"]
    assert top.harmonic_impropers.shape == (0, 4) and top.virtual_sites.shape == (0, 4)
    assert top.constraints is None and top.rigid_waters is None and not top.prepared
    # The view of the system equals that of the loaded data.
    system = loaded.make_system()
    for name in ("masses", "charges", "atomic_numbers", "bonds", "dihedrals", "residue_starts"):
        assert np.array_equal(getattr(system.topology, name), getattr(top, name)), name
    # The D213 arrays of System agree with the view.
    assert system.atom_names == top.atom_names and system.type_names == top.type_names
    assert np.array_equal(system.charges, top.charges)
    assert np.array_equal(system.particle_types, top.particle_types)
    assert np.array_equal(system.residue_indices, top.residue_indices)
    assert system.residue_names == [top.residue_names[r] for r in top.residue_indices]
    print(f"amber: {n} atoms, {top.residue_count} residues, {len(top.bonds)} bonds, "
          f"{len(top.angles)} angles, {len(top.dihedrals)} dihedrals equal the prmtop")
    return o


def gromacs():
    for defines in ([], ["FLEXIBLE"]):
        o = gromacs_oracle(inputs / "gromacs/system.top", defines)
        loaded = mdir.load_gromacs(str(inputs / "gromacs/system.top"),
                                   str(inputs / "gromacs/system.gro"), defines=defines)
        top = loaded.topology
        n = len(o["names"])
        check_common(top, n)
        assert top.atom_names == o["names"] and top.residue_names == o["residue_names"]
        assert np.array_equal(top.charges, o["charges"])
        assert np.array_equal(top.residue_starts, o["residue_starts"])
        assert rows(top.bonds) == o["bonds"], "bonds"
        assert rows(top.angles) == o["angles"], "angles"
        assert set(map(tuple, top.dihedrals.tolist())) == o["quads"], "dihedrals"
        assert set(top.atomic_numbers.tolist()) == {1, 6, 8}
        assert np.array_equal(top.masses > 2, top.atomic_numbers > 1)
        assert top.constraints is None and top.rigid_waters is None
        # The waters that SETTLE holds, after preparation.
        system, state = loaded.make_system(), loaded.make_state()
        system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.7
        system.rigid_water = not defines
        program = mdir.compile(system, state, mdir.Integrator(), mdir.Ensemble(),
                               mdir.Execution(), mdir.Schedule())
        prepared = program.topology
        assert prepared.prepared and prepared.constraints.shape == (0, 2)
        assert rows(prepared.rigid_waters) == rows(np.array(o["settles"]).reshape(-1, 3))
        print(f"gromacs {'+'.join(defines) or 'rigid'}: {n} atoms, {top.residue_count} residues, "
              f"{len(top.bonds)} bonds, {len(top.angles)} angles, "
              f"{len(set(map(tuple, top.dihedrals.tolist())))} dihedral quadruples in "
              f"{len(top.dihedrals)} terms, {len(prepared.rigid_waters)} rigid waters equal the files")


def charmm_view():
    o = psf_oracle(charmm / "toy.psf")
    loaded = mdir.load_charmm(str(charmm / "toy.psf"), str(charmm / "toy.crd"),
                              [str(charmm / "toy.rtf"), str(charmm / "toy.prm")])
    top = loaded.topology
    n = len(o["names"])
    check_common(top, n)
    assert top.atom_names == o["names"] and top.residue_names == o["residue_names"]
    assert np.array_equal(top.charges, o["charges"]) and np.array_equal(top.masses, o["masses"])
    assert np.array_equal(top.residue_starts, o["residue_starts"])
    assert [top.type_names[t] for t in top.particle_types] == o["types"]
    elements = {12.011: 6, 1.008: 1, 15.9994: 8, 22.9898: 11, 35.45: 17}
    assert top.atomic_numbers.tolist() == [elements[m] for m in o["masses"]]
    assert rows(top.bonds) == o["bonds"] and rows(top.angles) == o["angles"]
    proper = top.dihedrals[~top.improper_dihedrals]
    assert set(map(tuple, proper.tolist())) == o["quads"], "dihedrals"
    impropers = set(map(tuple, top.harmonic_impropers.tolist())) | set(
        map(tuple, top.dihedrals[top.improper_dihedrals].tolist()))
    assert impropers == o["impropers"], "impropers"
    print(f"charmm: {n} atoms, {top.residue_count} residues, {len(top.bonds)} bonds, "
          f"{len(top.angles)} angles, {len(o['quads'])} dihedral quadruples in "
          f"{len(top.dihedrals)} terms, {len(top.harmonic_impropers)} impropers equal the PSF")


# --- select against the control file's path -------------------------------

AMBER_INPUT = """[input]
topology = "{topology}"
coordinates = "{coordinates}"
"""
CHARMM_INPUT = """[input]
format = "CHARMM"
topology = "{charmm}/toy.psf"
coordinates = "{charmm}/toy.crd"
parameters = ["{charmm}/toy.rtf", "{charmm}/toy.prm"]
"""
REST = """[energy]
cutoff = 8.0
pairlist_distance = 9.0
electrostatics = "CUTOFF"
[dynamics]
time_step = 0.001
steps = 0
[ensemble]
ensemble = "NVE"
temperature = 300.0
[boundary]
type = "PERIODIC"
{box}
"""


def control_selections(text, name):
    path = work / f"{name}.toml"
    path.write_text(text)
    run = subprocess.run([model_test, str(path), "--selections"],
                         capture_output=True, text=True)
    if run.returncode != 0:
        return None, run.stderr
    chosen = {}
    for line in run.stdout.splitlines():
        what, _, values = line.partition(":")
        chosen[what] = np.array([int(v) for v in values.split()], dtype=np.int64)
    return chosen, ""


def probe(mask):
    """A restraint and an external term over `mask`."""
    return (f'[[restraints]]\nselection = "{mask}"\nforce_constant = 1.0\n'
            f'[[energy.external]]\nname = "probe"\nexpression = "k*x"\n'
            f'selection = "{mask}"\nk = 0.0\n')


def selections():
    cases = [
        ("amber", AMBER_INPUT.format(topology=inputs / "dipeptide/dipeptide.prmtop",
                                     coordinates=inputs / "dipeptide/dipeptide.inpcrd"), "",
         mdir.load_amber(str(inputs / "dipeptide/dipeptide.prmtop"),
                         str(inputs / "dipeptide/dipeptide.inpcrd")),
         ["!:WAT & !@H*", ":ALA", ":4-9@O", "@CA,C,N", ":NME | :ACE", "@H*",
          ":1-3 & !(@C,O | @N)", "@1-10,20-25", ":AL?", "*", "!*", ":XYZ", "@1000-1168 & :1"]),
        ("opc", AMBER_INPUT.format(topology=inputs / "opc/opc.prmtop",
                                   coordinates=inputs / "opc/opc.inpcrd"), "",
         mdir.load_amber(str(inputs / "opc/opc.prmtop"), str(inputs / "opc/opc.inpcrd")),
         [":WAT", "@EPW", ":1-3", "@O*"]),
        ("charmm", CHARMM_INPUT.format(charmm=charmm), "box = [32.0, 32.0, 32.0]",
         mdir.load_charmm(str(charmm / "toy.psf"), str(charmm / "toy.crd"),
                          [str(charmm / "toy.rtf"), str(charmm / "toy.prm")]),
         [":TOY", ":NAX | :CLX", "@C*", ":1@O3", ":2", "@H?"]),
    ]
    count = 0
    for name, head, box, loaded, masks in cases:
        top = loaded.topology
        massive = top.masses > 0
        for k, mask in enumerate(masks):
            chosen, error = control_selections(head + REST.format(box=box) + probe(mask),
                                               f"{name}-{k}")
            assert chosen is not None, (mask, error)
            mine = top.select(mask)
            assert mine.dtype == np.int64 and mine.ndim == 1 and not mine.flags.writeable
            assert np.all(np.diff(mine) > 0)
            assert np.array_equal(chosen["external probe"], mine), (name, mask)
            assert np.array_equal(chosen["restrained"], mine[massive[mine]]), (name, mask)
            count += 1
        if name == "opc":
            mine = top.select(":WAT")
            assert np.any(~massive[mine]), "opc has virtual sites"
            assert np.array_equal(top.select("@EPW"), top.virtual_sites[:, 0])
    # couple of [free_energy]: whole molecules, and nothing.
    fep = inputs / "fep"
    loaded = mdir.load_amber(str(fep / "eth_wat.prmtop"), str(fep / "eth_wat.inpcrd"))
    top = loaded.topology
    head = AMBER_INPUT.format(topology=fep / "eth_wat.prmtop", coordinates=fep / "eth_wat.inpcrd")
    for k, mask in enumerate([":LIG", ":1", "!:WAT", ":LIG | :2-3", ":XYZ"]):
        text = (head + REST.format(box="") + f'[free_energy]\ncouple = "{mask}"\nstate = 0\n'
                '[free_energy.lambdas]\ncoulomb = [0.0, 1.0]\nvdw = [0.0, 1.0]\n')
        chosen, error = control_selections(text, f"couple-{k}")
        mine = top.select(mask)
        if chosen is None:
            assert len(mine) == 0 and "selects no particle" in error, (mask, error)
        else:
            assert len(mine) > 0 and np.array_equal(chosen["couple"], mine), mask
        count += 1
    # A mask that does not parse: the control file's diagnostic.
    loaded = mdir.load_amber(str(inputs / "dipeptide/dipeptide.prmtop"),
                             str(inputs / "dipeptide/dipeptide.inpcrd"))
    head = AMBER_INPUT.format(topology=inputs / "dipeptide/dipeptide.prmtop",
                              coordinates=inputs / "dipeptide/dipeptide.inpcrd")
    for k, mask in enumerate([":1-", "@CA &", "(:1", ":0", ":ALA)", ":1,", "#5"]):
        chosen, error = control_selections(head + REST.format(box="") + probe(mask), f"bad-{k}")
        message = expect(mdir.InputError, lambda: loaded.topology.select(mask))
        assert chosen is None and message in error, (mask, message, error)
        count += 1
    # An empty mask, which a control file takes as no selection at all.
    expect(mdir.InputError, lambda: loaded.topology.select(""), "in the mask '', at 1")
    expect(TypeError, lambda: loaded.topology.select(1))
    assert issubclass(mdir.InputError, ValueError)
    print(f"select: {count} masks equal the control file's selections, "
          f"empty ones and diagnostics included")


# --- Constraints after preparation ----------------------------------------

def constraints(o):
    loaded = mdir.load_amber(str(inputs / "dipeptide/dipeptide.prmtop"),
                             str(inputs / "dipeptide/dipeptide.inpcrd"))
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.7
    system.rigid_hydrogen_bonds, system.rigid_water = True, True
    system.water_residues = ["WAT"]
    compile_ = lambda: mdir.compile(system, state, mdir.Integrator(), mdir.Ensemble(),
                                    mdir.Execution(), mdir.Schedule())
    program = compile_()
    top = program.topology
    assert top.prepared and top.constraints.dtype == np.int64 and top.rigid_waters.dtype == np.int64
    water = set(np.flatnonzero(np.array(o["residue_names"]) == "WAT").tolist())
    residue = o["residue_indices"]
    heavy_first = lambda i, j: (i, j) if o["atomic_numbers"][i] != 1 else (j, i)
    expected = [heavy_first(i, j) for i, j in o["hydrogen_bonds"] if residue[i] not in water]
    assert rows(top.constraints) == rows(expected), "constraints"
    expected_waters = [(s, s + 1, s + 2) for r, s in enumerate(o["residue_starts"]) if r in water]
    assert rows(top.rigid_waters) == rows(expected_waters), "rigid waters"
    # The degrees of freedom of the program, from the view (#112).
    dof = 3 * int((top.masses > 0).sum()) - len(top.constraints) - 3 * len(top.rigid_waters) - 3
    # Those of the engine: 2 K / (k_B T) of a state at the start.
    drawn = mdir.InitialState.draw_velocities(state, system, 300.0, 1)
    schedule = mdir.Schedule()
    schedule.steps = 0
    sim = mdir.Simulation(mdir.compile(system, drawn, mdir.Integrator(), mdir.Ensemble(),
                                       mdir.Execution(), schedule))
    sim.run(0, energy=True)
    energies = sim.state().energies
    engine = 2.0 * energies["kinetic"] / (0.0083144626181532 * energies["temperature"])
    assert abs(engine - dof) < 1e-6, (engine, dof)
    del sim
    print(f"constraints: {len(top.constraints)} bonds of hydrogen and "
          f"{len(top.rigid_waters)} rigid waters equal the prmtop; {dof} degrees of freedom, "
          f"as the engine counts")
    # A program's view does not follow the system once it is stale; a new
    # program's does.
    before = top.constraints
    system.rigid_hydrogen_bonds = False
    assert program.stale and np.array_equal(program.topology.constraints, before)
    assert compile_().topology.constraints.shape == (0, 2)
    print(f"bonds of the compiled topology: {len(top.bonds)} of {sum(o['bonds'].values())}")


# --- Copies and snapshots --------------------------------------------------

def snapshots():
    loaded = mdir.load_amber(str(inputs / "dipeptide/dipeptide.prmtop"),
                             str(inputs / "dipeptide/dipeptide.inpcrd"))
    system = loaded.make_system()
    view = system.topology
    masses, names, bonds = view.masses, view.atom_names, view.bonds
    # Changing what was copied out changes nothing.
    copied = np.array(masses)
    copied[:] = 0
    names[0] = "XX"
    assert view.masses[0] > 0 and view.atom_names[0] != "XX"
    assert system.topology.atom_names[0] != "XX"
    # Changing the system changes no view taken before.
    system.water_residues = ["XYZ"]
    system.rigid_water = True
    system.tuple_terms = [mdir.TupleTerm()]
    system.restraints = [mdir.Restraint(":ALA", 10.0)]
    assert np.array_equal(view.masses, masses) and np.array_equal(view.bonds, bonds)
    assert np.array_equal(system.topology.bonds, bonds)
    # Views have no setters.
    expect(AttributeError, lambda: setattr(view, "masses", masses))
    expect(AttributeError, lambda: setattr(system, "topology", view))
    # An update of a simulation's tunable charges reaches neither its
    # program's view nor the system's (D213).
    system = loaded.make_system()
    system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.7
    system.tunables = [mdir.Tunable("q", "charge")]
    state, schedule = loaded.make_state(), mdir.Schedule()
    schedule.steps = 0
    program = mdir.compile(system, state, mdir.Integrator(), mdir.Ensemble(),
                           mdir.Execution(), schedule)
    charges = program.topology.charges
    sim = mdir.Simulation(program)
    sim.tunables["q"] = 0.5 * charges
    sim.run(0, energy=True)
    assert np.array_equal(program.topology.charges, charges)
    assert np.array_equal(system.topology.charges, charges)
    assert np.array_equal(sim.program.topology.charges, charges)
    del sim, program
    # A view outlives what it came from.
    del system, loaded
    assert view.particle_count == len(masses) and repr(view).startswith("Topology(1168 particles")
    print("copies, dtypes, read-only views and snapshots passed")


o = amber()
gromacs()
charmm_view()
selections()
constraints(o)
snapshots()
print("topology views passed")
