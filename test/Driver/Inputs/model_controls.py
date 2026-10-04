"""Fixed explicit CLI inputs for native model parity in MD units."""
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
target = sys.argv[2]
for name in ("amber", "gromacs", "constraints", "triclinic", "nvt", "npt", "charmm", "charmm-triclinic"):
    amber = name in ("amber", "constraints", "nvt", "npt", "charmm", "charmm-triclinic")
    topology = "dipeptide/dipeptide.prmtop" if amber else "triclinic/water.top" if name == "triclinic" else "gromacs/system.top"
    coords = "dipeptide/dipeptide.inpcrd" if amber else "triclinic/dodecahedron.gro" if name == "triclinic" else "gromacs/system.gro"
    charmm = name.startswith('charmm')
    if charmm:
        topology, coords = 'charmm/toy.psf', 'charmm/toy.crd'
    input_extra = 'format = "CHARMM"\nparameters = ["charmm/toy.rtf", "charmm/toy.prm"]' if charmm else '' if amber else 'defines = ["FLEXIBLE"]'
    boundary_extra = 'box = [32.0,32.0,32.0,80.0,90.0,90.0]' if name == 'charmm-triclinic' else 'box = [32.0,32.0,32.0]' if charmm else ''
    for precision in ("DOUBLE", "MIXED"):
        pme = '[pme]\ngrid = [28,28,28]' if name == 'triclinic' else ''
        thermostat = '[thermostat]\nmethod = "V-RESCALE"\ninterval = 10' if name in ('nvt','npt') else ''
        barostat = '[barostat]\nmethod = "C-RESCALE"\ninterval = 10' if name == 'npt' else ''
        text = f'''[input]
topology = "{topology}"
coordinates = "{coords}"
{input_extra}
[energy]
cutoff = 8.0
pairlist_distance = 9.0
electrostatics = "{ 'PME' if name == 'triclinic' else 'CUTOFF' }"
{pme}
[constraints]
hydrogen_bonds = {'true' if name == 'constraints' else 'false'}
rigid_water = {'true' if name == 'constraints' else 'false'}
[dynamics]
time_step = 0.0005
steps = 20
[output]
energy_interval = 10
[ensemble]
ensemble = "{ 'NPT' if name == 'npt' else 'NVT' if name == 'nvt' else 'NVE' }"
temperature = 300.0
{thermostat}
{barostat}
[boundary]
type = "PERIODIC"
{boundary_extra}
[execution]
target = "{target}"
precision = "{precision}"
'''
        (root / f"{name}-{precision.lower()}.toml").write_text(text)
