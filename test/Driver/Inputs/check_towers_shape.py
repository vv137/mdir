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
        return
    assert result.returncode == 0, report
    return report["run"]["neighbors"]["towers_shape"]


for field in ("panel", "stripe"):
    check("{" + field + " = 33}", "multiple")
check('{threads=512, blocks_per_sm=1, panel=512, stripe=480, records=7680, cell_height="FULL"}',
      "shared memory per block")
check('{threads=512, blocks_per_sm=2, panel=320, stripe=352, records=3520, cell_height="HALF"}',
      "shared memory per SM")
check('{cell_height="FULL"}', "cell height",
      template.replace("34.7786, 34.7786, 34.7786", "34.7786, 34.7786, 21.0"))
shape = check('{threads=512, blocks_per_sm=2, panel=352, stripe=384, records=1536, split=1, cell_height="HALF"}')
assert shape["requested"] == shape["effective"]
assert set(shape["source"].values()) == {"control"}
explicit = check('{cell_height="FULL"}', text=template + '\nneighbor_structure = "TOWERS"')
assert explicit["effective"]["cell_height"] == "FULL"
check('"AUTO"', "mixed precision", template.replace('"MIXED"', '"DOUBLE"') + '\nneighbor_structure = "TOWERS"')
partial = check('{split=2, cell_height="FULL"}')
assert partial["effective"]["threads"] == 512 and partial["effective"]["split"] == 2
assert partial["source"]["threads"] == "AUTO" and partial["source"]["split"] == "control"

# Feed the two explicit preflight layouts to the generated rectangular entry.
sys.path.insert(0, str(root.parents[1] / "test" / "Runtime" / "Inputs"))
from towers_exclusions import evaluate
import towers_rebuild

towers_rebuild.MDIR_OPT = str(Path(mdir).with_name("mdir-opt"))

work = control.with_suffix(".kernels")
work.mkdir(exist_ok=True)
outputs = []
for workers, panel, stripe, records, split in [(32, 32, 32, 1, 1), (64, 64, 64, 2, 2)]:
    settings = (f'{{threads={workers}, panel={panel}, stripe={stripe}, records={records}, '
                f'blocks_per_sm=1, split={split}, cell_height="FULL"}}')
    effective = check(settings)["effective"]
    outputs.append(evaluate(work, True, effective["threads"], effective))
assert outputs[0] == outputs[1]
