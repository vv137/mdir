#!/usr/bin/env bash
# Says which step of a script failed, where its log is, and the first lines
# of the log that name an error, on standard error (#654):
#
#   scripts/release/report-failure.sh SCRIPT STEP STATUS [LOG]
#
# publish.sh calls it from its trap on ERR. Without LOG, or with a log that
# does not exist, the step's own output is above. The lines are the first
# ten that start with `FAILED:` (ninja), contain `error:` (compilers,
# Python), or contain `CMake Error` (CMake's configure), the last with the
# two lines of its message after it (#662).
set -uo pipefail
script=${1:?usage: $0 SCRIPT STEP STATUS [LOG]}
step=${2:?usage: $0 SCRIPT STEP STATUS [LOG]}
status=${3:?usage: $0 SCRIPT STEP STATUS [LOG]}
log=${4:-}
{
  if [[ -n $log && -f $log ]]; then
    echo "$script: the step '$step' failed (status $status); its log: $log"
    # The first ten lines that name an error; a `CMake Error at FILE:LINE`
    # line is followed by its message, the next two lines (#662).
    lines=$(awk '
      n >= 10 { exit }
      after > 0 { print; after--; next }
      /CMake Error/ { print; n++; after = 2; next }
      /^FAILED:|error:/ { print; n++ }
    ' "$log")
    if [[ -n $lines ]]; then
      echo "$script: the first errors of the log:"
      printf '%s\n' "$lines" | sed 's/^/  /'
    else
      echo "$script: the log names no error; its last lines:"
      tail -n 10 "$log" | sed 's/^/  /'
    fi
  else
    echo "$script: the step '$step' failed (status $status); its output is above"
  fi
} >&2
