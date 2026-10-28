"""The initial restraint column uses the energy its correction defines (#706).

The metadata-only plugin refuses evaluation: emitting a GPU target needs no
device or optional model framework. Every emitted whole module must verify.
"""
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys

root, place = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
mdir, printer = sys.argv[3:5]
place.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location(
    "hosted_order", root / "scripts/builder-equivalence/order.py")
order = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = order
spec.loader.exec_module(order)
controls, _ = order.controls(root, place)
library = order.metadata_plugin(root, place, os.environ.get("CC", "cc"))
env = dict(os.environ, MDIR_MLIP_PLUGIN=str(library), MDIR_MLIP_PAIRS="exact")
ssa = r"%[a-zA-Z0-9_.$-]+"
checked = verified = 0
for control in controls:
    if not control.name.startswith("model-host-restraints-"):
        continue
    source = control.read_text()
    assert source.count("[execution]\n") == 1
    control.write_text(source.replace("[execution]\n", '[execution]\ntarget = "GPU"\n'))
    for segments in ([], ["--segments"]):
        emitted = subprocess.run(
            [mdir, "emit", "--stage=module", *segments, str(control)],
            env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False)
        assert emitted.returncode == 0, emitted.stderr
        text = emitted.stdout
        calls = list(re.finditer(r"func.call @mdrtWriteTerms\((" + ssa + r")\)", text))
        assert calls, "missing initial energy-column report"
        for call in calls:
            # Check data flow, independent of the correction's SSA spelling.
            start = text.rfind("func.func @", 0, call.start())
            prefix = text[max(0, start):call.start()]
            cast = re.search(re.escape(call[1]) + r"\s*=\s*memref.cast\s+(" + ssa + r")\s*:", prefix)
            assert cast, "missing energy-column buffer cast"
            stores = list(re.finditer(
                r"memref.store\s+(" + ssa + r"),\s*" + re.escape(cast[1]) + r"\[", prefix))
            assert stores, "missing stored energy column"
            energy = stores[-1][1]
            assert re.search(re.escape(energy) + r"\s*=\s*vector.reduction\s+<add>",
                             prefix[:stores[-1].start()]), (
                "stored restraint energy has no defining reduction: " + energy)
        # Resolve SSA and verify every operation in the complete module.
        result = subprocess.run(
            [printer, "-", "-o", os.devnull], input=text, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert result.returncode == 0, result.stderr
        verified += 1
        checked += 1
assert checked == verified == 12, (checked, verified)
print(f"hosted restraint stores: {checked} parsed and bound to their defining reduction")
print(f"whole modules: {verified} verified")
