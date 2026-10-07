"""Topology.to_openmm (D[python-topology]): the OpenMM topology of a view
against those of OpenMM's own readers of the same files, and the error
without OpenMM.

Usage: python_topology_openmm.py INPUTS CHARMM MODE, with INPUTS
test/Driver/Inputs, CHARMM the files of charmm.test, and MODE `openmm` or
`hidden`."""
import pathlib
import sys

inputs, charmm, mode = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]

if mode == "hidden":
    # A None entry makes `import openmm` fail.
    sys.modules["openmm"] = None
    import mdir
    loaded = mdir.load_amber(str(inputs / "dipeptide/dipeptide.prmtop"),
                             str(inputs / "dipeptide/dipeptide.inpcrd"))
    try:
        loaded.topology.to_openmm()
    except ImportError as exc:
        assert "Topology.to_openmm needs OpenMM" in str(exc), str(exc)
        print("to_openmm without openmm: ImportError")
    else:
        raise AssertionError("expected ImportError")
    sys.exit(0)

# OpenMM before MDIR, whose libraries need the newer C++ runtime.
import openmm.app as app
import openmm.unit as unit
import numpy as np
import mdir


def bonds(topology):
    return {tuple(sorted((a.index, b.index))) for a, b in topology.bonds()}


def residues(topology):
    return [[a.index for a in r.atoms()] for r in topology.residues()]


def compare(name, mine, ref, view):
    elements = [a.element for a in mine.atoms()]
    assert elements == [a.element for a in ref.atoms()], name
    assert [a.name for a in mine.atoms()] == view.atom_names, name
    assert [r.name for r in mine.residues()] == view.residue_names, name
    assert residues(mine) == residues(ref), name
    assert bonds(mine) == bonds(ref), name
    # Chains hold whole molecules, as many as the bonds make.
    for bond in mine.bonds():
        assert bond[0].residue.chain is bond[1].residue.chain, name
    return f"{mine.getNumAtoms()} atoms, {mine.getNumResidues()} residues, " \
           f"{len(bonds(mine))} bonds, {mine.getNumChains()} chains"


def read_molecules(path):
    """The number of molecules of a prmtop, ATOMS_PER_MOLECULE."""
    lines = pathlib.Path(path).read_text().splitlines()
    k = next(i for i, l in enumerate(lines) if l.startswith("%FLAG ATOMS_PER_MOLECULE")) + 2
    count = 0
    while not lines[k].startswith("%"):
        count += len(lines[k].split())
        k += 1
    return count


for name in ("dipeptide/dipeptide", "opc/opc", "fep/eth_wat"):
    loaded = mdir.load_amber(str(inputs / f"{name}.prmtop"), str(inputs / f"{name}.inpcrd"))
    view, state = loaded.topology, loaded.make_state()
    mine = view.to_openmm(state.cell)
    ref = app.AmberPrmtopFile(str(inputs / f"{name}.prmtop")).topology
    text = compare(name, mine, ref, view)
    assert mine.getNumChains() == read_molecules(inputs / f"{name}.prmtop"), name
    box = mine.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
    assert np.array_equal(np.array([list(v) for v in box]), state.cell.vectors), name
    assert np.allclose(np.array([list(v) for v in box]),
                       np.array([list(v) for v in ref.getPeriodicBoxVectors().value_in_unit(unit.nanometer)]),
                       rtol=1e-12, atol=0), name
    assert view.to_openmm().getPeriodicBoxVectors() is None
    print(f"to_openmm {name.split('/')[0]}: {text} equal AmberPrmtopFile's")

# OpenMM's GromacsTopFile does not read the test's topology (bonds without a
# function number), so the rigid waters, whose O-H bonds come from SETTLE,
# are checked against the flexible ones, whose bonds the file gives.
gromacs = {}
for defines in ([], ["FLEXIBLE"]):
    top = inputs / "gromacs/system.top"
    loaded = mdir.load_gromacs(str(top), str(inputs / "gromacs/system.gro"), defines=defines)
    gromacs[bool(defines)] = loaded.topology.to_openmm()
assert bonds(gromacs[False]) == bonds(gromacs[True])
assert residues(gromacs[False]) == residues(gromacs[True])
assert gromacs[False].getNumChains() == 4 + 60
print(f"to_openmm gromacs: {len(bonds(gromacs[False]))} bonds with SETTLE equal those of "
      f"FLEXIBLE, {gromacs[False].getNumChains()} chains")

loaded = mdir.load_charmm(str(charmm / "toy.psf"), str(charmm / "toy.crd"),
                          [str(charmm / "toy.rtf"), str(charmm / "toy.prm")])
ref = app.CharmmPsfFile(str(charmm / "toy.psf")).topology
text = compare("charmm", loaded.topology.to_openmm(), ref, loaded.topology)
print(f"to_openmm charmm: {text} equal CharmmPsfFile's")
print("to_openmm passed")
