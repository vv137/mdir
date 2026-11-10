"""Serialize archived native initial search state; no force-state claim or GPU use."""
import argparse
import copy
import hashlib
import importlib.util
import json
import pathlib
import struct
import tomllib
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('archive', type=pathlib.Path)
parser.add_argument('w1', type=pathlib.Path)
parser.add_argument('destination', type=pathlib.Path)
parser.add_argument('independent_provenance', type=pathlib.Path)
args = parser.parse_args()
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()
spec = importlib.util.spec_from_file_location('archived_decode', args.archive / 'row-clocks-decode.py')
decoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decoder)
provenance = json.loads(args.independent_provenance.read_text())
args.destination.mkdir(parents=True, exist_ok=True)
records = {}
for system in ['ubq-opc', 'ubq-tip3p', 'opc-small', 'opc-large', 'tip3p-small', 'tip3p-large']:
    directory = args.archive / system
    stem = directory / 'row-clocks-0001-attempt-1'
    geometry = pathlib.Path(str(stem) + '.geometry.bin')
    entry_file = pathlib.Path(str(stem) + '.entries.bin')
    row_file = pathlib.Path(str(stem) + '.rows.bin')
    g = decoder.load_geometry(geometry)
    h = g['header']
    assert h['serial'] == h['attempt'] == 1 and h['flags'] == 0
    eraw = entry_file.read_bytes()
    assert decoder.header(eraw) == h
    entries = np.frombuffer(eraw, dtype=decoder.ENTRY, offset=128)
    assert len(entries) == h['entries'] and not entries['reserved'].any()
    W, A, WG, AG = (h[k] for k in ['waters', 'atoms', 'water_groups', 'atom_groups'])
    used = [WG > 0, AG > 0, AG > 0]
    cell = g['cells'][0, :3].copy()
    reaches = np.zeros(3, dtype='<f8')
    for tag, active in enumerate(used):
        if active:
            assert np.array_equal(g['cells'][tag, :3].view('<u8'), cell.view('<u8'))
            assert (g['cells'][tag, 3:] == 0).all(), 'not diagonal'
            reaches[tag] = g['reaches'][0, tag]
            assert reaches[tag] > 0 and np.isfinite(reaches[tag])
    assert (cell > 0).all() and np.isfinite(cell).all()
    indexes = np.r_[np.arange(WG), W + np.arange(AG)]
    starts = g['groups'][0, indexes].astype('<i4')
    lengths = g['groups'][1, indexes].astype('<i4')
    table = np.empty((len(entries), 6), dtype='<i4')
    table[:, 0] = np.where(entries['row'] < WG, 0, np.where(entries['row'] < WG + AG, 1, 2))
    for column, name in enumerate(['row', 'partner', 'image', 'mask', 'reserved'], 1):
        table[:, column] = entries[name]
    header = struct.pack('<10Q', 0x31544147544f5250, 1, W, A, WG, AG, len(entries),
                         len(g['graph_neighbors']), h['chunks'], h['unit_capacity'])
    payload = b''.join([header, cell.astype('<f8').tobytes(), reaches.tobytes(),
                        g['points'].astype('<f8').tobytes(), g['order'].astype('<i4').tobytes(),
                        starts.tobytes(), lengths.tobytes(), g['graph_rows'].astype('<i4').tobytes(),
                        g['graph_neighbors'].astype('<i4').tobytes(), table.tobytes()])
    output = args.destination / (system + '.gate')
    output.write_bytes(payload)
    control = directory / 'control.toml'
    reference_control = args.w1 / 'job-a' / system / 'on' / 'control.toml'
    diagnostic = tomllib.loads(control.read_text())
    native = tomllib.loads(reference_control.read_text())
    def starting_state(c):
        c = copy.deepcopy(c)
        c.pop('output', None)
        c['dynamics'].pop('steps', None)
        c['dynamics'].pop('rate', None)
        return c
    assert starting_state(diagnostic) == starting_state(native), (system, 'starting control differs')
    input_hashes = {key: {'path': value, 'sha256': sha(pathlib.Path(value))}
                    for key, value in diagnostic['input'].items()}
    records[system] = dict(actual_snapshot_head='f9ea87ef8788807dab9924b970faee23391b3b7d',
        comparison_source_contract='retained W1 5ea35cdb; no fresh W1 snapshot claim',
        selected_serial=1, selected_attempt=1, native_header=h,
        source_files={str(p): sha(p) for p in [geometry, entry_file, row_file]},
        gate_file=str(output), gate_sha256=sha(output),
        control_sha256=sha(control), W1_control_sha256=sha(reference_control),
        unchanged_starting_fields=starting_state(native), input_hashes=input_hashes,
        cell_bits=cell.view('<u8').tolist(), active_outer_reach_bits=reaches.view('<u8').tolist(),
        reference_bits_sha256=hashlib.sha256(g['points'].tobytes()).hexdigest(),
        native_group_basis_sha256=hashlib.sha256(g['order'].tobytes()+starts.tobytes()+lengths.tobytes()).hexdigest(),
        unused_metadata='Unused SW/SS cells/reaches and all inner slots ignored, not asserted finite',
        force_state_provenance=False)
(args.destination / 'native-snapshot-manifest.json').write_text(json.dumps(dict(
    systems=records, independent_provenance=provenance,
    prototype_input_conversion='float((nativeFraction * nativeCell_nm) * 10.0); f32GeoL=float(nativeCell_nm*10.0); R2=float((nativeReach_nm*10.0)^2)',
    force_state_warning='No original particle membership/site-frame identity is inferred'), indent=2))
print('Prepared six archived f9 initial-entry states, exact used bits/control/input provenance; no GPU')
