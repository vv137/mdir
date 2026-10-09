#!/usr/bin/env python3
"""The inputs of test/Driver/Inputs/fep as a GROMACS topology and
coordinates, written once with ParmEd for both MDIR and GROMACS.

    gromacs_inputs.py WORK

Needs ParmEd (the Python of AmberTools). WORK receives eth_wat.top and
eth_wat.gro, the coordinates with seven decimals of nm.
"""
import os
import sys

import parmed

here = os.path.dirname(os.path.abspath(__file__))
inputs = os.path.join(here, '..', '..', '..', 'test', 'Driver', 'Inputs',
                      'fep')
work = sys.argv[1]
os.makedirs(work, exist_ok=True)
system = parmed.load_file(os.path.join(inputs, 'eth_wat.prmtop'),
                          os.path.join(inputs, 'eth_wat.inpcrd'))
system.save(os.path.join(work, 'eth_wat.top'), overwrite=True)
system.save(os.path.join(work, 'eth_wat.gro'), overwrite=True, precision=7)
