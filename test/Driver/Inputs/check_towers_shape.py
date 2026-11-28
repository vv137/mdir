"""Exercise manual tower geometry and device memory diagnostics through preflight."""

import json
from pathlib import Path
import subprocess
import sys

mdir, root, directory = sys.argv[1:]
root = Path(root).resolve() / "examples" / "argon"
control = Path(directory)
template = (root / "argon.toml").read_text().replace(
    '"argon.pdb"', json.dumps(str(root / "argon.pdb"))
).replace('"CPU"', '"GPU"').replace('"DOUBLE"', '"MIXED"')


def check(shape, reason="", text=template):
    control.write_text(text + "\ntowers_shape = " + shape + "\n")
    result = subprocess.run([mdir, "check", "--json", control], capture_output=True, text=True)
    report = json.loads(result.stdout)
    if reason:
        assert result.returncode == 1 and reason in report["errors"][0], report
        return report["errors"][0]
    assert result.returncode == 0, report
    return report["run"]["neighbors"]["towers_shape"]


sys.path.insert(0, str(root.parents[1] / "test" / "Runtime" / "Inputs"))
from towers_rebuild import device_capability

check('{screen="OTHER"}', "screen")
scalar = check('{screen="SCALAR", threads=128, blocks_per_sm=1}')
assert scalar["effective"]["screen"] == "SCALAR" and scalar["screen_reason"] == "control"
for field in ("panel", "stripe"):
    check("{" + field + " = 33}", "multiple")
check('{threads=512, blocks_per_sm=1, panel=512, stripe=480, records=7680, cell_height="FULL"}',
      "shared memory per block")
check('{threads=512, blocks_per_sm=2, panel=320, stripe=352, records=3520, cell_height="HALF"}',
      "shared memory per SM")
check('{cell_height="FULL"}', "cell height",
      template.replace("34.7786, 34.7786, 34.7786", "34.7786, 34.7786, 21.0"))
shape = check('{threads=512, blocks_per_sm=2, panel=352, stripe=384, records=1536, split=1, pack="COMPACT", cell_height="HALF"}')
assert shape["source"].pop("screen") == shape["requested"].pop("screen") == "AUTO"
assert shape["effective"].pop("screen") == ("MMA" if device_capability() >= 75 else "SCALAR")
assert shape["requested"] == shape["effective"]
assert set(shape["source"].values()) == {"control"}
diagnostic = check('{threads=384, blocks_per_sm=4, panel=32, stripe=32, records=1, cell_height="FULL"}',
                   "registers per thread", template + '\nneighbor_structure = "TOWERS"')
setup = subprocess.run([mdir, "run", control], capture_output=True, text=True)
assert setup.returncode and diagnostic in setup.stderr, (diagnostic, setup.stderr)
print(diagnostic)
explicit = check('{cell_height="FULL"}', text=template + '\nneighbor_structure = "TOWERS"')
assert explicit["effective"]["cell_height"] == "FULL"
check('"AUTO"', "mixed precision", template.replace('"MIXED"', '"DOUBLE"') + '\nneighbor_structure = "TOWERS"')
for split in (2, 4, 8):
    partial = check(f'{{split={split}, cell_height="FULL"}}')
    assert partial["effective"]["threads"] == 512 and partial["effective"]["split"] == split
    assert partial["source"]["threads"] == "AUTO" and partial["source"]["split"] == "control"
    assert partial["candidate_cells"] == [{"cell_height": "FULL", "split": split}]
for split in (3, 6, 16):
    check(f"{{split={split}}}", "split must be 1, 2, 4, or 8")

# Feed the two explicit preflight layouts to the generated rectangular entry.
from towers_exclusions import evaluate
import towers_rebuild

towers_rebuild.MDIR_OPT = str(Path(mdir).with_name("mdir-opt"))

work = control.with_suffix(".kernels")
work.mkdir(exist_ok=True)
for height, reach in [("FULL", 1), ("HALF", 2)]:
    outputs = []
    for workers, panel, stripe, records, residency, split in [
        (32, 32, 32, 1, 1, 1), (64, 64, 64, 2, 2, 2)
    ]:
        settings = (f'{{threads={workers}, panel={panel}, stripe={stripe}, records={records}, '
                    f'blocks_per_sm={residency}, split={split}, cell_height="{height}"}}')
        effective = check(settings)["effective"]
        outputs.append(evaluate(work, True, effective["threads"], effective, reach))
    assert outputs[0] == outputs[1]
