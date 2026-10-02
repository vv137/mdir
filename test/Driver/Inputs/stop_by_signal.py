# Starts `mdir run` on a control file and sends it a signal once its log has
# the row of a step, and the signal again each time the run says that it
# will stop, `count` times in all. Prints the output of the run and how it
# ended. If it stopped at a checkpoint, runs the control file again without
# a stop, for the steps up to that checkpoint, and compares the two states.
#
#   python stop_by_signal.py <mdir> <control> <step> <signal> [<count>]

import os
import re
import signal
import subprocess
import sys

mdir, control, step, name = sys.argv[1:5]
count = int(sys.argv[5]) if len(sys.argv) > 5 else 1
kind = getattr(signal, name)

run = subprocess.Popen([mdir, "run", control], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, bufsize=1)
sent = 0
stopped = None
for line in run.stdout:
    sys.stdout.write(line)
    if sent == 0:
        row = re.match(r"INFO:\s+(\d+)\s", line)
        if row and int(row.group(1)) >= int(step):
            run.send_signal(kind)
            sent = 1
    elif sent < count and line.startswith("mdir: stopping at the next"):
        run.send_signal(kind)
        sent += 1
    found = re.match(r"MDIR: stopped after step (\d+) on \S+; '(.*)' holds",
                     line)
    if found:
        stopped = int(found.group(1)), found.group(2)
status = run.wait()
sys.stdout.flush()
if status < 0:
    print(f"ended by {signal.Signals(-status).name}")
else:
    print(f"exit status {status}")
if stopped is None:
    sys.exit(0)

# The same run, not interrupted, up to the step of the checkpoint.
at, path = stopped
with open(control) as file:
    text = file.read()
text = re.sub(r"(?m)^steps\s*=.*$", f"steps = {at}", text)
text = text.replace(os.path.basename(path), "upto.h5")
upto = control.replace(".toml", ".upto.toml")
with open(upto, "w") as file:
    file.write(text)
subprocess.run([mdir, "run", upto], stdout=subprocess.DEVNULL, check=True)
compared = subprocess.run(
    [mdir, "checkpoint", path, os.path.join(os.path.dirname(path), "upto.h5")],
    capture_output=True, text=True)
print(f"against {at} steps not interrupted: {compared.stdout.strip()}")
