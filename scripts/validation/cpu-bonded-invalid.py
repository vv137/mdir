#!/usr/bin/env python3
"""Require coordinated rejection of malformed topology input."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    ap=argparse.ArgumentParser();ap.add_argument('executable');args=ap.parse_args()
    env=dict(os.environ);env.pop('DISPLAY',None)
    cases=['101 101 1 1\n','101 999 1 1\n','101 108 1 1\n108 101 1 1\n',
           '101 108 0 1\n','101 108 1 -1\n','101 108 nan 1\n',
           '101 108 1e300 1\n','101 108 1\n101 108 1 1\n',
           '101 108 1 1 junk\n']
    with tempfile.TemporaryDirectory(prefix='mdir-bond-invalid-') as temp:
        snapshot,bonds=Path(temp)/'snapshot',Path(temp)/'bonds'
        snapshot.write_text('2 12 12 12 2.5 1 1\n101 1 3 3\n108 8 3 3\n')
        for bad in cases:
            bonds.write_text(bad)
            run=subprocess.run(['mpiexec','--oversubscribe','-n','2',args.executable,str(snapshot),
                                f'--bonds={bonds}','--precision=mixed'],text=True,
                                capture_output=True,env=env,timeout=30)
            assert run.returncode != 0 and 'mdir-cpu-lj:' in run.stderr,(bad,run.stdout,run.stderr)
    print('PASS',len(cases),'invalid bond inputs')


if __name__=='__main__': main()
