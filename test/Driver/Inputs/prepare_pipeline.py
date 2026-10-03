"""Shorten the printed pipeline for a CPU test, keeping its checkpoint chain."""

import pathlib
import re
import subprocess
import sys

mdir, examples, directory = sys.argv[1:]
examples = pathlib.Path(examples).resolve()
directory = pathlib.Path(directory)
directory.mkdir(parents=True, exist_ok=True)
for stage in ("minimize", "nvt", "npt", "production"):
    text = subprocess.check_output([mdir, "template", stage], text=True)
    text = text.replace("system.prmtop", str(examples / "ala3.prmtop"))
    text = text.replace("system.inpcrd", str(examples / "ala3.inpcrd"))
    text = re.sub(
        r"(?m)^(steps|energy_interval|trajectory_interval|checkpoint_interval)"
        r"\s*=\s*\d+",
        r"\1 = 10",
        text,
    )
    text = text.replace('"GPU"', '"CPU"').replace('"MIXED"', '"DOUBLE"')
    (directory / (stage + ".toml")).write_text(text)
