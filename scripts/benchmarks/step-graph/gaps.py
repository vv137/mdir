"""Partition ordinary-step idle in a node-level Nsight Systems SQLite export.

For launch traces pass the comma-separated kernel ordinals anchoring each kind
of step in the 600-step, energy-100, coupling-10 timing protocol.
"""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
from itertools import accumulate
import json
from pathlib import Path
import re
import sqlite3


def group(name):
    if name.startswith(("memcpy", "Memcpy")):
        return "copy"
    if "integration" in name:
        return "integration"
    if "pme_" in name or "fft" in name:
        return "reciprocal"
    if "pair_for" in name or "towers_" in name:
        return "direct"
    if "tuple_for" in name or "particle_for" in name:
        return "other force"
    return name


def analyze(path, anchors):
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    ks = db.execute("""select k.start,k.end,s.value,k.streamId from CUPTI_ACTIVITY_KIND_KERNEL k
        join StringIds s on s.id=k.shortName order by k.start""").fetchall()
    events = list(ks)
    for kind in ("MEMCPY", "MEMSET"):
        table = "CUPTI_ACTIVITY_KIND_" + kind
        if db.execute("select 1 from sqlite_master where name=?", (table,)).fetchone():
            label = ("case copyKind when 2 then 'readback' when 1 then 'upload' else 'Memcpy' end"
                     if kind == "MEMCPY" else "'Memset'")
            events.extend(db.execute(f"select start,end,{label},streamId from {table}"))
    events.sort()
    starts = [ev[0] for ev in events]
    latest_ends = list(accumulate((ev[1] for ev in events), max))
    if anchors:
        indices = [i for i, (_, _, name, _) in enumerate(ks)
                   if re.fullmatch(r"mdir_run_particle_for_integration\d+_(\d+)", name)
                   and int(name.rsplit("_", 1)[1]) in anchors]
        assert len(indices) == 600
        windows = [(ks[indices[n-1]][0], ks[indices[n]][0], "integration", "integration")
                   for n in range(150, 600) if n % 10]
    else:
        ends = [i for i, k in enumerate(ks) if k[2] in ("finishStep", "finishHybrid")]
        assert len(ends) == 540
        windows = [(ks[i][1], ks[j][1], ks[i][2], ks[j][2])
                   for i,j in zip(ends[135:], ends[136:])
                   if all(k[2] != "acceptedBoundary" for k in ks[i+1:j])]
    gaps, durations, streams = Counter(), Counter(), set()
    for begin, end, prev, last in windows:
        frontier = begin
        # A kernel on another trace lane can start just before the marker
        # ends and occupy much of the interval. Include every crossing.
        lo = bisect_right(latest_ends, begin)
        hi = bisect_left(starts, end)
        for start, stop, name, stream in events[lo:hi]:
            start, stop = max(start, begin), min(stop, end)
            if stop <= start:
                continue
            dest = group(name)
            streams.add(stream)
            gaps[prev + " -> " + dest] += max(0, start-frontier)
            durations[dest] += max(0, min(stop,end)-max(frontier,start))
            if stop > frontier:
                frontier, prev = stop, dest
        gaps[prev + " -> " + last] += max(0, end-frontier)
    apis = dict(db.execute("""select s.value,count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r
        join StringIds s on s.id=r.nameId where s.value like '%GraphInstantiate%'
        or s.value like '%GraphUpload%' or s.value like '%GraphLaunch%' group by s.value"""))
    def means(counter):
        return {k: v / (1000*len(windows)) for k,v in counter.most_common() if v}
    return dict(samples=len(windows), streams=sorted(streams), graph_apis=apis,
                idle_us=sum(gaps.values())/(1000*len(windows)), gap_us=means(gaps),
                busy_us=means(durations))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--anchors", type=lambda s: set(map(int, s.split(","))))
    args = parser.parse_args()
    print(json.dumps(analyze(args.trace, args.anchors), indent=2))
