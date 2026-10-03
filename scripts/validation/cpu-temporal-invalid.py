#!/usr/bin/env python3
"""Coordinated rejection checks for the experimental temporal MPI driver."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('executable')
    args = parser.parse_args()
    env = dict(os.environ)
    env.pop('DISPLAY',None)
    count = 0
    with tempfile.TemporaryDirectory(prefix='mdir-temporal-invalid-') as temp:
        snapshot,state = Path(temp)/'snapshot',Path(temp)/'state'
        snapshot.write_text('2 12 12 12 2.5 1 1\n101 5.4 3 3\n108 6.6 3 3\n')
        valid = '101 1 0 0 0\n108 1.7 0 0 0\n'
        base = [args.executable,str(snapshot),'--steps=1','--dt=.001',f'--state={state}']
        def reject(command):
            nonlocal count
            run = subprocess.run(command,text=True,capture_output=True,env=env,timeout=30)
            assert run.returncode != 0, (command,run.stdout,run.stderr)
            assert 'mdir-cpu-lj:' in run.stderr, (command,run.stderr)
            count += 1
        for bad in ('101 1 0 0 0\n', '101 1 0 0 0\n101 1 0 0 0\n',
                    '101 1 0 0 0\n999 1 0 0 0\n',
                    '101 0 0 0 0\n108 1 0 0 0\n',
                    '101 1 nan 0 0\n108 1 0 0 0\n', valid+'junk\n'):
            state.write_text(bad)
            reject(['mpiexec','--oversubscribe','-n','2']+base)
        state.write_text(valid)
        for extra in ('--dt=0','--dt=nan','--skin=-1','--repeat=2','--emit=dist'):
            reject(['mpiexec','--oversubscribe','-n','2']+base+[extra])
        # MPMD launch: inconsistent control flow must abort, not strand a peer
        # in a payload collective or an option-dependent early return.
        reject(['mpiexec','--oversubscribe','-n','1']+base+['--halo=sync',':','-n','1']+
               base+['--halo=async'])
    print('PASS',count,'invalid inputs and participant mismatch checks')


if __name__ == '__main__':
    main()
