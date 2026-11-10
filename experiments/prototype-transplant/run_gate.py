"""One authorized six-state initial-entry gate; stop on the first difference.

This runner never submits a job. Invoke it through the required GPU helper
only after the root-reviewed command manifest's authorization flag is true.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('root', type=pathlib.Path)
args = parser.parse_args()
root = args.root.resolve()
manifest = json.loads((root / 'gate-command-manifest.json').read_text())
assert manifest['authorized_for_submission'] is True, 'STOP: root gate authorization is FALSE'

def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()

for path, digest in manifest['frozen_files'].items():
    assert sha(path) == digest, ('changed frozen input/source/binary', path)
results = []
for case in manifest['cases']:
    directory = pathlib.Path(case['directory'])
    assert directory.is_dir() and not (directory / 'run.log').exists(), 'STOP: no overwrite/rerun'
    print('INITIAL ENTRY BEGIN', case['system'], flush=True)
    with (directory / 'run.log').open('x') as output:
        status = subprocess.run(case['command'], cwd=directory,
                                stdout=output, stderr=subprocess.STDOUT).returncode
    row = dict(system=case['system'], exit=status)
    report = directory / 'result.json'
    if report.exists():
        row['comparison'] = json.loads(report.read_text())
    results.append(row)
    (root / 'execution-results.json').write_text(json.dumps(results, indent=2))
    print('INITIAL ENTRY END', json.dumps(row), flush=True)
    if status or not row.get('comparison', {}).get('pass'):
        raise SystemExit('STOP first refusal/entry difference; all raw evidence retained; no later cases or retry')
print('Six initial-entry gates passed; no trajectory/timing/force-state conclusion', flush=True)
