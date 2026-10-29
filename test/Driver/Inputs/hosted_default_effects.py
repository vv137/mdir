"""Verify hosted dynamics/minimization effect summaries without a device (#707)."""
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
    "hosted_effect_order", root / "scripts/builder-equivalence/order.py")
order = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = order
spec.loader.exec_module(order)
_, controls = order.controls(root, place)
# Keep this fixed 72-module proof independent of later order-matrix additions.
controls = [path for path in controls if re.fullmatch(
    r"model-host-(?:nve|npt|min)-(?:DOUBLE|MIXED)\.toml", path.name)]
assert len(controls) == 6, controls
library = order.metadata_plugin(root, place, os.environ.get("CC", "cc"))
checked = 0
for mode in ("exact", "padded", "search"):
    env = dict(os.environ, MDIR_MLIP_PLUGIN=str(library), MDIR_MLIP_PAIRS=mode)
    for original in controls:
        source = original.read_text()
        assert "[[restraints]]" not in source
        assert source.count("[execution]\n") == 1
        source = source.replace("[execution]\n", '[execution]\ntarget = "GPU"\n')
        for constrained in (False, True):
            text = source
            if constrained:
                # Hosted models currently require flexible particles. Check
                # the independent constraint-stop declaration on classical
                # controls, without changing that support restriction.
                text, removed = re.subn(
                    r"(?ms)^\[\[energy\.model\]\]\n.*?(?=^\[|\Z)", "", text)
                assert removed == 1
                assert "hydrogen_bonds = false" in text
                text = text.replace("hydrogen_bonds = false", "hydrogen_bonds = true")
            control = place / (original.stem + ("-constrained" if constrained else "-free") + ".toml")
            control.write_text(text)
            for segments in ([], ["--segments"]):
                result = subprocess.run(
                    [mdir, "emit", "--stage=module", *segments, str(control)],
                    env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                assert result.returncode == 0, result.stderr
                module = result.stdout
                verified = subprocess.run(
                    [printer, "-", "-o", os.devnull], input=module, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                assert verified.returncode == 0, verified.stderr
                assert ("mdrt.host_call @mdrtModelEvaluate" in module) == (not constrained)
                programs = list(re.finditer(
                    r"dyn\.program @([^\s(]+)[\s\S]*?attributes\s*\{([\s\S]*?)\}\s*\{", module))
                assert programs, "missing hosted program"
                for program in programs:
                    for effect in ("read", "write"):
                        resources = re.search(effect + r"\s*=\s*\[([^]]*)\]", program[2])
                        if not constrained:
                            assert resources and '"<Default>"' in resources[1], program[0]
                        else:
                            assert not resources or '"<Default>"' not in resources[1], program[0]
                        if constrained and program[1] != "descend":
                            assert '"dyn.stop"' in resources[1], program[0]
                checked += 1
assert checked == 72, checked
print("effect modules: 72 verified (36 hosted, 36 classical constrained)")
print("exact/padded/search; double/mixed; NVE/NPT/min; run/segments")
