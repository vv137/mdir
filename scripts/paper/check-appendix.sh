#!/bin/bash
# Checks that the example of Appendix A of the paper is what
# `mdir template amber` prints, so that the manual follows the code.
#
#   scripts/paper/check-appendix.sh MDIR
set -e
mdir=$1
paper=$(cd "$(dirname "$0")/../../docs/paper" && pwd)
diff <(sed -n '/^```toml/,/^```/p' "$paper/A-control-file.md" | sed '1d;$d') \
     <("$mdir" template amber) && echo "Appendix A agrees with mdir template amber"
