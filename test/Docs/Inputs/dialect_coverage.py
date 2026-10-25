"""Every op that docs/ops.md names is in the generated reference of its
dialect (docs/dialects/, #610), and ops.md links each reference. A name of
the dialect that is not an op, a discardable attribute such as
`mdrt.segment`, is in the reference when its dialect's TableGen
description names it."""

from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
dialects = ["md", "dyn", "md_exec", "md_dist", "mdrt"]
reference, texts = {}, {}
for dialect in dialects:
    text = (root / "docs" / "dialects" / f"{dialect}.md").read_text(encoding="utf-8")
    texts[dialect] = text
    reference[dialect] = set(re.findall(r"^### `(" + dialect + r"\.[a-z0-9_.]+)`",
                                        text, re.M))
ops = (root / "docs" / "ops.md").read_text(encoding="utf-8")
# An op is `dialect.name` in code, a span or a fenced block, not a type (!)
# or an attribute (#).
code, fence = [], False
for line in ops.split("\n"):
    if line.lstrip().startswith("```"):
        fence = not fence
        continue
    if fence:
        code.append(line)
    else:
        code += [m.group(2) for m in re.finditer(r"(`+)(.+?)\1", line)]
named = set()
for span in code:
    for m in re.finditer(r"(?<![!#\w.])(" + "|".join(dialects) + r")\.([a-z][a-z0-9_]*)\b",
                         span):
        named.add((m.group(1), m.group(0)))
missing = sorted(op for dialect, op in named
                 if op not in reference[dialect] and f"`{op}`" not in texts[dialect])
attributes = sum(1 for dialect, op in named if op not in reference[dialect])
for op in missing:
    print(f"docs/ops.md names {op}, which docs/dialects/ does not have")
unlinked = [d for d in dialects if f"dialects/{d}.md" not in ops]
for dialect in unlinked:
    print(f"docs/ops.md does not link dialects/{dialect}.md")
total = sum(len(v) for v in reference.values())
print(f"{len(named)} names in ops.md ({len(named) - attributes} ops, "
      f"{attributes} attributes), {total} ops in the reference, "
      f"{len(missing)} missing, {len(unlinked)} references unlinked")
sys.exit(1 if missing or unlinked else 0)
