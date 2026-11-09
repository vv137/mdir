"""Check initializer payload provenance, rather than just presence of renumber.

Both dynamic identity/nonidentity cases use the same lowered function. Device
renumbering's host result is a file-order placeholder; initialization must copy
its registered device payload, for waters and for exclusions.
"""
import re
import subprocess
import sys

opt, fixture = sys.argv[1:]
for device in (True, False):
    options = 'kept-tuples=true device-tuples=true' if device else 'device-tuples=false kept-tuples=false'
    text = subprocess.check_output([opt, fixture, '--convert-md-exec-to-gpu=' + options], text=True)
    body = text.split('  func.func @exercise(', 1)[1].split('  func.func @main(', 1)[0]
    lines = [line.strip() for line in body.splitlines()
             if line.startswith('    ') and not line.startswith('     ')]
    definitions = {}
    registry = set()
    copies = {}
    for line in lines:
        match = re.match(r'(%[\w.$]+) = (.*)', line)
        if match:
            definitions[match[1]] = match[2]
            if 'call @mdrtDeviceTuplesRenumbered(' in line:
                registry.add(match[1])
        if 'gpu.memcpy' in line:
            values = re.findall(r'%[\w.$]+', line)
            copies[values[-2]] = values[-1]

    def root(value):
        seen = set()
        while value not in seen:
            seen.add(value)
            operation = definitions.get(value, '')
            match = re.match(r'memref\.(?:cast|memory_space_cast) (%[\w.$]+)', operation)
            if not match:
                return value
            value = match[1]
        raise AssertionError('cyclic cast provenance')

    calls = [line for line in lines if 'call @mdrtTemplatePairsCreateMembers(' in line]
    assert len(calls) == 1, calls
    arguments = re.search(r'CreateMembers\(([^)]*)\)', calls[0])[1].split(', ')
    payloads = [root(arguments[0]), root(arguments[3])]
    for payload in payloads:
        assert definitions[payload].startswith('memref.alloc'), 'initializer retained a file-order host payload: ' + definitions[payload]
        if device:
            assert payload in copies, 'registered tuple payload was not copied to host'
            assert root(copies[payload]) in registry, 'initializer copied a file-order placeholder'
        else:
            assert payload not in copies, 'host-renumbered payload should already be on the host'
    assert bool(registry) == device
    print(('device' if device else 'host') + ': molecule and exclusion payload provenance passed')
