"""Exercise the exact diagnostic allowlist independently of either builder."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

spec = importlib.util.spec_from_file_location('equivalence', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
key = ('extra/static.toml', 'cpu segments', 'module', 'printed')
old = (1, b'', b'<generated>: error: operand #0 does not dominate this use\n')
new = (1, b'', module.MESSAGE.encode())
entry = dict(zip(('case', 'target', 'stage', 'form'), key))
for name, value in (('old', old), ('new', new)):
    entry[name] = dict(zip(('status', 'stdout', 'stderr'),
                          (value[0], value[1].decode(), value[2].decode())))
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'expected.json'
    def load(entries):
        path.write_text(json.dumps(dict(version=1, issue=721, entries=entries)))
        return module.ExpectedErrors(path)
    errors = load([entry])
    assert not errors.complete()
    for index in range(4):
        wrong = list(key)
        wrong[index] += '-other'
        assert not errors.accept(tuple(wrong), old, new)
    for before, after in ((old, (0, b'', new[2])),
                          (old, (1, b'changed', new[2])),
                          (old, (1, b'', new[2] + b'note\n')),
                          ((1, b'', old[2] + b'note\n'), new),
                          (old, old),
                          ((0, b'old program', b''), (0, b'new program', b''))):
        assert not errors.accept(key, before, after)
    assert errors.accept(key, old, new)
    assert errors.complete()
    hashes = tuple([str(old[0])] + [hashlib.sha256(v).hexdigest() for v in old[1:]])
    errors = load([entry])
    assert not errors.accept(key, ('1', 'wrong', hashes[2]), new, hashes=True)
    assert errors.accept(key, hashes, new, hashes=True)
    assert errors.complete()
    for entries in ([entry, entry], [dict(entry, target='cpu')],
                    [dict(entry, new=dict(status=0, stdout='', stderr=module.MESSAGE))],
                    [dict(entry, new=dict(status=1, stdout='', stderr='other error\n'))]):
        try:
            load(entries)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid exception accepted')
print('exact diagnostic exception checks passed')
