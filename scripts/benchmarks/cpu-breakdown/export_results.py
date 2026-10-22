"""Copy benchmark JSONL into a path-free record suitable for the repository.

    python export_results.py SOURCE/results.jsonl DEST.jsonl

The source files can contain local paths and host allocation details. Only
portable measurement fields are retained; no input is evaluated.
"""

import json
import sys
from pathlib import Path

from timing import gromacs_step_ms, pmemd_all_step_ms


FIELDS = (
    "system", "sites", "variant", "round", "tag", "rate_ms", "steps",
    "threads", "precision", "exact", "reach", "ensemble", "has_velocities",
    "builds", "build_interval", "wall_s", "ok", "grid", "cutoff",
    "rate_source",
)
REQUIRED = {"system", "variant", "round", "tag", "rate_ms", "steps", "ok",
            "grid", "cutoff"}


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: export_results.py SOURCE.jsonl DEST.jsonl")
    source, destination = map(Path, sys.argv[1:])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open() as src, destination.open("w") as dst:
        for line in src:
            raw = json.loads(line)
            if REQUIRED - raw.keys():
                raise ValueError(f"record missing fields: {sorted(REQUIRED - raw.keys())}")
            record = {key: raw[key] for key in FIELDS if key in raw}
            if record["variant"] == "p1" and record["ok"]:
                record["rate_ms"] = pmemd_all_step_ms(
                    (Path(raw["dir"]) / "mdout").read_text())
                record["rate_source"] = "pmemd all-steps Per Step(ms)"
            elif record["variant"][0] in "gx" and record["ok"]:
                record["rate_ms"] = gromacs_step_ms(
                    (Path(raw["dir"]) / "md.log").read_text())
                record["rate_source"] = "GROMACS Performance ms/step"
            dst.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
