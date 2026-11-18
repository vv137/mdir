"""segments_ptx.py MODULE LOWERED: the structural facts of the segmented SIMT
schedule (D421, the mechanism table of D417, Section 8,
S3), from the module that `mdir emit` prints and the PTX of its lowered
program (MDIR_GPU_BINARY=ptx).

  - The constants of a slice inline (E3): the lookups of the tables in each
    window of a segment; a segment of two taken kinds has none.
  - No load of a stored extra point: the loads of 64 bits (the f64
    positions) of the kernel that packs the frames after a build, three a
    member of each shape packed, and of the writer of each step
    (D[md-alg-segments-writer]), the stored site of a kind not among them.
  - The rotation of the partners by shuffles, and no instruction for a site
    pair outside the table: for each kernel of the schedule, its index
    shuffles and its guards of the cutoff of a site pair (setp.geu.f32,
    the branch around a pair that is not within it).
"""
import re
import sys

module, lowered = open(sys.argv[1]).read(), open(sys.argv[2]).read()

# The windows of the segments of the first potential (@energy), in order.
energy = module[module.index('md.potential @energy'):]
energy = energy[:energy.index('md.potential @', 10)]
for match in re.finditer(r'md\.sum_segment_window [^\n]*segment\((\d+), (\d+)\)', energy):
    start = match.end()
    depth, k = 1, energy.index('{', start) + 1
    while depth:
        depth += {'{': 1, '}': -1}.get(energy[k], 0)
        k += 1
    print(f'segment ({match.group(1)}, {match.group(2)}): '
          f'{energy[start:k].count("md.lookup")} lookups')

kernels = []
for m in re.finditer(r'gpu\.binary @(\S+)\s+\[#gpu\.object<[^>]*>, properties = \{[^}]*\}, '
                     r'assembly = "((?:[^"\\]|\\.)*)"', lowered):
    ptx = re.sub(r'\\([0-9A-Fa-f]{2})', lambda h: chr(int(h.group(1), 16)), m.group(2))
    kernels.append((m.group(1), ptx))
schedule = [(n, p) for n, p in kernels if n.startswith('mdir_run_pair_for')]
# The packing: the kernels that load the f64 positions and narrow them, a
# thread a place (the refresh's after a build, D[md-alg-segments-writer]),
# and the writer of the frames of each step, a thread a member.
packers = schedule + [(n, p) for n, p in kernels
                      if n.startswith('mdir_run_refresh_segmented_pairs')]
packs = sorted({len(re.findall(r'\bld\.global\.b64\b', p)) for n, p in packers
                if 'shfl' not in p and re.search(r'\bcvt\.rn\.f32\.f64\b', p)})
print('packing: ' + ', '.join(f'{k} loads of 64 bits' for k in packs))
writers = sorted({len(re.findall(r'\bld\.global\.b64\b', p)) for n, p in kernels
                  if n.startswith('mdir_run_write_unit_frames')})
print('writer: ' + ', '.join(f'{k} loads of 64 bits' for k in writers))
counts = sorted({(len(re.findall(r'\bshfl\.sync\.idx\.b32\b', p)),
                  len(re.findall(r'\bsetp\.geu\.f32\b', p))) for n, p in schedule
                 if re.search(r'\bshfl\.sync\.idx\.b32\b', p) and re.search(r'\bsetp\.geu\.f32\b', p)})
for shuffles, guards in counts:
    print(f'schedule: {shuffles} index shuffles, {guards} guards of site pairs')
