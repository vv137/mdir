"""Reads the output of the sampler (sampler.c).

    prof.py DIR [--program CONTROL | --map MAP] [--top N] [--json OUT]
    prof.py --classify CONTROL|MAP

DIR is the SAMPLER_OUT of a run. Each parallel region of the program
(mdir_run..omp_par.N, mdrt.pme_spread_..omp_par) is given a part from the
profile map of the program (D[profile-map], docs/profile-map.md), which
the compiler writes: with --map, the map of the run (`[output]
profile_map`); with --program, the control file of the run is emitted
(`$MDIR emit CONTROL --stage=pipeline --profile-map=FILE`), whose compiled
code is that of the run, region for region. The map names the op that
each region comes from and the set it runs over; the part follows from
them alone (`part_of_region`). Without a map, the regions are not
attributed to parts: there is no fallback that guesses from the code.

Prints, for the samples of the main thread's wall clock and for those of the
CPU time of all threads: the share by leaf symbol, and by "owner", the first
frame of the stack that is compiled code of the program (a function of the
JIT) together with whether the leaf is that code or a library it called.

--classify prints the part of each region of the map of CONTROL (or of the
map MAP) and its record, one line each, and reads no samples
(test/Lit/profile-map.test).
"""
import bisect
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict

d = sys.argv[1]
classify_only = d == "--classify"
top = 40
jout = None
regions = {}
profile_map = None
program = None
a = sys.argv[2:]
if classify_only:
    if len(a) != 1:
        sys.exit("prof.py --classify CONTROL|MAP")
    a = ["--map" if a[0].endswith(".json") else "--program", a[0]]
while a:
    if a[0] == "--top":
        top = int(a[1]); a = a[2:]
    elif a[0] == "--map":
        profile_map = json.load(open(a[1])); a = a[2:]
    elif a[0] == "--program":
        program = a[1]; a = a[2:]
    elif a[0] == "--json":
        jout = a[1]; a = a[2:]
    else:
        sys.exit("unknown " + a[0])

PATTERNS = [
    # The parallel regions of the templates (mdrt.pme_spread_..omp_par.N)
    # are named after the function they are outlined from.
    (r"^mdrt\.pme_(spread|zero|real)", "PME spread"),
    (r"^mdrt\.pme_(convolve|scale|tables|influence|point_sums)", "PME convolution"),
    (r"^mdrt\.pme_(gather|potential|splines)", "PME gather"),
    (r"^mdrt\.pme_fft", "PME FFT"),
    # The B-splines and places, where the frame of their caller is not
    # seen: those of the spreading and the gathering together.
    (r"^mdrt\.pme_(bspline|raise|place|spline|window|clenshaw|inverse_cell)", "PME splines"),
    (r"^(_mlir_ciface_)?mdrtFFT|^(pass|rad[fb]|cfft|rfft|pass_all|copy_and_norm|cfftp|rfftp)", "PME FFT"),
    (r"^mdrt\.(build_neighbors|spatial_order|minimum_image|cell_|wrap)|^(_mlir_ciface_)?mdrtHostMatrix|^mdrtCountBuild",
     "list builds"),
    (r"^mdrt(Tuples|Write|Finish|AddBath|Bussi|CheckSpread)|^_mlir_ciface_mdrt", "runtime calls"),
]

# The parts of the loops over tuples, by their tuple set; the other sets
# (bonds, angles, dihedrals, pairs14, urey_bradley, impropers, cmaps, ...)
# are of the bonded terms. The sets are named by the builder.
TUPLE_PARTS = [
    (r"^excluded$", "excluded pairs"),
    (r"^(settles|shake\d+)$", "constraints"),
    (r"^anchors$", "particles"),
]
# The parts of the other ops of the lowering.
SOURCE_PARTS = {
    "pair_for": "pairs",
    "particle_for": "particles",
    "scatter_add": "particles",
    "refresh_neighbors": "list builds",
    "spatial_order": "list builds",
    "permute": "list builds",
}


def part_of_region(record):
    """The part of a region of the profile map, from its record alone."""
    source = record.get("source")
    if source is None:
        return "unmapped"
    if source == "tuple_for":
        for pattern, part in TUPLE_PARTS:
            if re.search(pattern, record.get("set", "")):
                return part
        return "bonded"
    if source == "template":
        for pattern, part in PATTERNS:
            if re.search(pattern, record.get("template", "")):
                return part
        return "other: " + record.get("template", "")
    return SOURCE_PARTS.get(source, "other: " + source)


if program:
    with tempfile.TemporaryDirectory() as scratch:
        path = os.path.join(scratch, "profile-map.json")
        subprocess.run([os.environ.get("MDIR", "mdir"), "emit", os.path.abspath(program), "--stage=pipeline",
                        "--profile-map=" + path],
                       cwd=os.path.dirname(os.path.abspath(program)), stdout=subprocess.DEVNULL, check=True)
        profile_map = json.load(open(path))
if profile_map is not None:
    for record in profile_map["regions"]:
        if record.get("function"):
            regions[record["function"]] = part_of_region(record)
if classify_only:
    for record in profile_map["regions"]:
        name = record.get("function") or "(no function)"
        detail = " ".join(f"{key}={record[key]}" for key in ("source", "stage", "set", "template")
                          if record.get(key))
        if record.get("potentials"):
            detail += " potentials=" + ",".join(record["potentials"])
        if record.get("terms"):
            detail += " terms=" + ",".join(record["terms"])
        print(f"{name}\t{part_of_region(record)}\t{detail} precision={record.get('precision') or '-'}")
    sys.exit(0)

DEPTH = 10
REC = struct.Struct("<IIQII" + "Q" * DEPTH)


def elf_symbols(data, loaded):
    """(address, size, name) of the functions of an ELF64 file. `loaded`: the
    addresses are sh_addr + st_value (a relocatable object of the JIT whose
    sections carry their load addresses); otherwise st_value."""
    if data[:4] != b"\x7fELF":
        return []
    e_type = struct.unpack_from("<H", data, 16)[0]
    shoff = struct.unpack_from("<Q", data, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    secs = []
    for i in range(shnum):
        off = shoff + i * shentsize
        name, typ, flags, addr, offset, size, link, info, align, entsize = struct.unpack_from(
            "<IIQQQQIIQQ", data, off)
        secs.append((name, typ, flags, addr, offset, size, link, info, align, entsize))
    out = []
    for s in secs:
        if s[1] not in (2, 11):  # SYMTAB, DYNSYM
            continue
        strtab = secs[s[6]]
        stroff = strtab[4]
        n = s[5] // 24
        for k in range(n):
            st_name, st_info, st_other, st_shndx, st_value, st_size = struct.unpack_from(
                "<IBBHQQ", data, s[4] + k * 24)
            if st_info & 0xF != 2 or st_shndx == 0 or st_shndx >= 0xFF00:
                continue
            end = data.index(b"\0", stroff + st_name)
            name = data[stroff + st_name:end].decode(errors="replace")
            addr = (secs[st_shndx][3] + st_value) if loaded else st_value
            out.append((addr, st_size, name))
    return out, e_type


class Symbolizer:
    def __init__(self, d):
        self.ranges = []  # (start, end, name, origin)
        for f in sorted(os.listdir(d)):
            if re.match(r"jit-\d+\.o$", f):
                data = open(os.path.join(d, f), "rb").read()
                syms, _ = elf_symbols(data, True)
                for addr, size, name in syms:
                    self.ranges.append((addr, addr + max(size, 1), name, "jit"))
        self.maps = []
        for line in open(os.path.join(d, "maps-exit.txt" if os.path.exists(os.path.join(d, "maps-exit.txt"))
                                      else "maps.txt")):
            p = line.split()
            if len(p) < 6 or not p[5].startswith("/"):
                continue
            lo, hi = (int(x, 16) for x in p[0].split("-"))
            self.maps.append((lo, hi, int(p[2], 16), p[5]))
        self.maps.sort()
        self.files = {}
        self.ranges.sort()
        self.starts = [r[0] for r in self.ranges]

    def file_syms(self, path):
        if path not in self.files:
            try:
                data = open(path, "rb").read()
                syms, e_type = elf_symbols(data, False)
            except Exception:
                syms, e_type = [], 0
            syms = sorted(syms)
            base = 0
            if e_type == 3:  # shared object: the mapping of offset 0
                base = min(lo - off for lo, hi, off, p in self.maps if p == path)
            self.files[path] = (syms, [s[0] for s in syms], base)
        return self.files[path]

    def __call__(self, pc):
        i = bisect.bisect_right(self.starts, pc) - 1
        if i >= 0 and self.ranges[i][0] <= pc < self.ranges[i][1]:
            return self.ranges[i][2], "jit"
        for lo, hi, off, path in self.maps:
            if lo <= pc < hi:
                syms, starts, base = self.file_syms(path)
                v = pc - base
                k = bisect.bisect_right(starts, v) - 1
                lib = os.path.basename(path)
                if k >= 0 and v < syms[k][0] + max(syms[k][1], 1) + 4096:
                    return syms[k][2], lib
                return "?", lib
        return "?", "?"


sym = Symbolizer(d)
raw = open(os.path.join(d, "samples.bin"), "rb").read()
cache = {}


def name(pc):
    if pc not in cache:
        cache[pc] = sym(pc)
    return cache[pc]


recs = [REC.unpack_from(raw, k * REC.size) for k in range(len(raw) // REC.size)]
# The window: the second half of the time that the main thread spends in
# compiled code (the steps), as the rate of a run is that of its second half.
jit_times = [r[2] for r in recs if r[0] == 1 and any(name(pc)[1] == "jit" for pc in r[5:5 + r[3]])]
lo_t = (jit_times[0] + jit_times[-1]) / 2 if jit_times else 0
hi_t = jit_times[-1] if jit_times else 1 << 63


def part_of(frames):
    """The part of a stack (leaf first): the innermost frame that names one,
    the B-splines and places taking the part of their caller where it is
    seen."""
    shared = None
    for n, o in frames:
        if n in regions:
            return regions[n]
        for pat, part in PATTERNS:
            if re.search(pat, n):
                if part != "PME splines":
                    return part
                shared = shared or part
                break
    if shared:
        return shared
    for n, o in frames:
        if o == "jit":
            return "serial code of the entry" if n == "mdir_run" else "other: " + n
    return "outside the program: " + frames[0][0]


wall_part, cpu_part = Counter(), Counter()
wall_leaf, cpu_leaf = Counter(), Counter()
wall_owner, cpu_owner = Counter(), Counter()
tids = Counter()
times = []
for rec in recs:
    kind, tid, t, depth = rec[0], rec[1], rec[2], rec[3]
    if not lo_t <= t <= hi_t:
        continue
    pcs = rec[5:5 + depth]
    leaf, origin = name(pcs[0])
    owner = None
    for j, pc in enumerate(pcs):
        n, o = name(pc - (1 if j else 0))
        if o == "jit":
            owner = n + ("" if j == 0 else " <- " + leaf if origin != "jit" else "")
            break
    if owner is None:
        owner = f"[{origin}] {leaf}"
    frames = [name(pc - (1 if j else 0)) for j, pc in enumerate(pcs)]
    part = part_of(frames)
    if origin == "libomp.so" and kind == 2 and not any(o == "jit" for n, o in frames):
        part = "idle (threads waiting for work)"
    elif origin == "libomp.so" and kind == 1 and part == "serial code of the entry":
        part = "fork and join"
    if kind == 1:
        wall_part[part] += 1
    else:
        cpu_part[part] += 1
    if kind == 1:
        wall_leaf[f"{leaf} [{origin}]"] += 1
        wall_owner[owner] += 1
        times.append(t)
    else:
        cpu_leaf[f"{leaf} [{origin}]"] += 1
        cpu_owner[owner] += 1
        tids[tid] += 1


def show(title, c):
    total = sum(c.values())
    print(f"\n== {title}: {total} samples")
    for k, v in c.most_common(top):
        print(f"{100.0 * v / total:6.2f}%  {v:7d}  {k}")


show("wall of the main thread, by part", wall_part)
show("CPU of all threads, by part", cpu_part)
show("wall of the main thread, by owner", wall_owner)
show("wall of the main thread, by leaf", wall_leaf)
show("CPU of all threads, by owner", cpu_owner)
show("CPU of all threads, by leaf", cpu_leaf)
print(f"\nthreads with CPU samples: {len(tids)}; span of the wall samples "
      f"{(times[-1] - times[0]) / 1e9 if times else 0:.1f} s")
if jout:
    json.dump(dict(wall_part=wall_part, cpu_part=cpu_part, wall_owner=wall_owner, wall_leaf=wall_leaf, cpu_owner=cpu_owner, cpu_leaf=cpu_leaf),
              open(jout, "w"), indent=1)
