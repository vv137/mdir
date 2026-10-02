# Runs a command and prints its exit status after what it printed, so that
# a test can check a status other than 0 and 1.
#
#   python exit_status.py <command> [<argument> ...]

import subprocess
import sys

status = subprocess.run(sys.argv[1:]).returncode
sys.stdout.flush()
print(f"exit status {status}")
