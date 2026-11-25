"""The smallest measured LAUNCH winner exercises the inclusive size rule."""
import json
from pathlib import Path
import re
import subprocess
import sys

from manifest_schema import validate

suite, work = map(Path, sys.argv[1:3])
mdir = sys.argv[3]
inputs = suite / "Amber24_Benchmark_Suite/PME/Cellulose_production_NVE"
template = Path(__file__).with_name("step-graph.toml.in").read_text()
deck = template.format(inputs=inputs, mode="AUTO", period=10, steps=10)
deck = deck.replace("dipeptide.prmtop", "prmtop").replace("dipeptide.inpcrd", "inpcrd")
deck = deck.replace("[32, 32, 32]", "[256, 128, 128]")
deck = re.sub(r"^checkpoint.*\n", "", deck, flags=re.M)
deck = re.sub(r"^towers_shape = .*", 'towers_shape = {threads=512, blocks_per_sm=2, '
              'stripe=384, panel=256, records=2350, split=1, cell_height="FULL"}', deck, flags=re.M)
(work / "control.toml").write_text(deck)


def invoke(*args):
    return subprocess.check_output([mdir, *args], cwd=work, text=True)


choice = json.loads(invoke("check", "--json", "control.toml"))["run"]["step_graph"]
assert choice["requested"] == "AUTO" and choice["value"] == "LAUNCH", choice
assert choice["category"] == "rule" and choice["reason"], choice
assert choice["inputs"] == {"particles": 408609, "threshold": 408609}, choice
assert "paired_gain_us" not in choice and "setup_ms" not in choice
log = invoke("run", "control.toml")
record = json.loads((work / "choices.jsonl").read_text().splitlines()[0])
validate(record, json.loads((Path(__file__).resolve().parents[3] / "docs/run-manifest.schema.json").read_text()))
assert record["resolved_choices"]["step_graph"] == choice
assert json.loads(re.search(r"^MDIR: step_graph (.+)$", log, re.M)[1]) == choice
assert 'step_graph = "LAUNCH"' in invoke("pin", "choices.jsonl")
