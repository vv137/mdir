"""Sleeps, and writes the interval in which it ran to <log>/<name>."""
import os
import sys
import time

log, name = sys.argv[1], sys.argv[2]
start = time.monotonic()
time.sleep(1.0)
with open(os.path.join(log, name + ".interval"), "w") as file:
    file.write(f"{start} {time.monotonic()}\n")
