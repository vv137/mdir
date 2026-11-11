"""Test-only exact serialization/instrumentation; no production accessors."""
import argparse
import json
from pathlib import Path
import re

p = argparse.ArgumentParser()
p.add_argument('mode', choices=['instrument', 'compare'])
p.add_argument('files', nargs='+', type=Path)
p.add_argument('--expect-publications', choices=['fused', 'unfused', 'separate'])
a = p.parse_args()
if a.mode == 'instrument':
    src, dst = a.files
    text = src.read_text()
    start = text.index('func.func @evaluate(')
    next_function = re.search(r'\n[ \t]*func\.func ', text[start + 1:])
    assert next_function is not None, 'evaluate must have a following fixture function'
    end = start + 1 + next_function.start()
    body = text[start:end]
    assert a.expect_publications is not None, 'instrument requires an explicit fusion expectation'
    publication_launches = re.findall(
        r'gpu\.launch[^\n]*function\(@(evaluate_particle_for_integration1_template_publication_[0-9]+)\)', body)
    # lowerIntegration emits one function-entry across-count clear and two
    # work launches for each of the six marked predictor/constraint/writer
    # runs. Count actual launch attributes, not filenames or helper calls.
    expected = 18 if a.expect_publications == 'fused' else 0
    # Mixed families fuse the update without publication; a kernel of its
    # own follows each of the six runs and publishes the stored positions.
    writers = re.findall(
        r'gpu\.launch[^\n]*function\(@(evaluate_particle_for_integration2_template_writer_[0-9]+)\)', body)
    assert len(writers) == (6 if a.expect_publications == 'separate' else 0), (
        f'unexpected separate writer launches: {writers}')
    assert len(publication_launches) == expected, (
        f'expected {expected} actual publication integration launches, got {len(publication_launches)}')
    all_publication_launches = re.findall(
        r'gpu\.launch[^\n]*function\(@([^)]*_template_publication[^)]*)\)', body)
    assert len(all_publication_launches) == expected, 'unexpected publication integration shape'
    owners = re.findall(r'(%[\w.$]+) = memref.alloca\(\) : memref<5xi1>', body)
    assert len(owners) == 1, f'expected one template validity owner, got {owners}'
    assert body.count('call @r1_snapshot(') == 6
    body = body.replace('call @r1_snapshot(',
                        f'func.call @r1_dump_validity({owners[0]}) : (memref<5xi1>) -> ()\n  call @r1_snapshot(')
    text = text[:start] + body + text[end:]
    helper = '''
func.func @r1_dump_validity(%flags: memref<5xi1>) {
  %tag = arith.constant -795110 : i64
  %len = arith.constant 5 : i64
  func.call @printI64(%tag) : (i64) -> ()
  func.call @printNewline() : () -> ()
  func.call @printI64(%len) : (i64) -> ()
  func.call @printNewline() : () -> ()
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  scf.for %i = %c0 to %c5 step %c1 {
    %flag = memref.load %flags[%i] : memref<5xi1>
    %wide = arith.extui %flag : i1 to i64
    func.call @printI64(%wide) : (i64) -> ()
    func.call @printNewline() : () -> ()
  }
  return
}
'''
    # Input mdir-opt prints a builtin.module; never append outside it.
    close = text.rfind('}')
    assert text[close + 1:].strip() == ''
    dst.write_text(text[:close] + helper + text[close:])
else:
    fused, standalone, reference = a.files
    def read(path):
        words = [int(x) for x in path.read_text().split()]
        energies, records = [], []
        i = 0
        while i < len(words):
            tag = words[i]
            i += 1
            if -795110 <= tag <= -795100:
                length = words[i]
                i += 1
                assert length > 0 and i + length <= len(words)
                records.append((tag, words[i:i + length]))
                i += length
            else:
                energies.append(tag)
        assert len(energies) == 7
        # Five flags followed by positions, seven frame slots, packed frames,
        # and tuple failure, repeated for six post-writer/pre-refresh phases.
        assert [r[0] for r in records] == ([-795110] + list(range(-795100, -795110, -1))) * 6
        assert energies == json.loads(reference.read_text())
        return records
    x, y = read(fused), read(standalone)
    for i, (left, right) in enumerate(zip(x, y)):
        assert left == right, f'exact publication mismatch at record {i}, tag {left[0]}'
    print('7 independent exact energies per mode; 66 exact publication records agree')
