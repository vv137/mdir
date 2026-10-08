#!/bin/bash
# Checks that the example of Appendix A of the paper is what
# `mdir template amber` prints, so that the manual follows the code.
#
#   scripts/paper/check-appendix.sh MDIR
#
# MDIR is the `mdir` to ask: a path, or a name on the PATH. The script
# exits with 0 if they agree, with 1 and the difference if they do not, and
# with 2 and its usage if it was not given one `mdir` that can be run.
set -e
if [ $# -ne 1 ] || ! command -v "$1" > /dev/null; then
  [ $# -ne 1 ] || echo "$0: '$1' is not a program that can be run" >&2
  echo "usage: $0 MDIR" >&2
  echo "  MDIR  the mdir whose 'template amber' Appendix A is compared with" >&2
  exit 2
fi
mdir=$1
paper=$(cd "$(dirname "$0")/../../docs/paper" && pwd)
diff <(sed -n '/^```toml/,/^```/p' "$paper/A-control-file.md" | sed '1d;$d') \
     <("$mdir" template amber) && echo "Appendix A agrees with mdir template amber"
