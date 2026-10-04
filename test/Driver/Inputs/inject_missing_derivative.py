"""Replace a named scalar expression op to test source attribution."""
import re
import sys
from pathlib import Path
source = Path(sys.argv[1]).read_text()
pattern = r'(\s*%[\w]+\s*=\s*)(?:arith.addf|arith.subf|arith.mulf|arith.divf)\s+(%[\w]+),\s+(%[\w]+)\s*:\s*f64\s*(loc\("control-file term:[^"\n]+"\))'
source, count = re.subn(pattern,
    lambda m: f'{m[1]}"test.no_derivative"({m[2]}, {m[3]}) : (f64, f64) -> f64 {m[4]}',
    source, count=1)
if count != 1:
    raise SystemExit("no named scalar expression found")
Path(sys.argv[2]).write_text(source)
