#!/usr/bin/env python3
"""Links the citations of the paper and writes its list of references.

    scripts/paper/build-references.py [docs/paper]

Each section of the paper cites an entry of docs/references.md as [[Key]].
The script makes every such citation a link, [[Key]](references.md#key),
and writes docs/paper/references.md with the entries cited, in the order
of their keys, without the notes on where MDIR uses them. It stops on a
key that docs/references.md does not have, and on an entry marked
UNVERIFIED; it lists the cited entries without a DOI, which
verify-references.py cannot check against a registry.
"""
import glob
import os
import re
import sys

CITE = re.compile(r"\[\[([A-Za-z0-9]+)\]\](\(references\.md#[a-z0-9]+\))?")


def main():
    paper = sys.argv[1] if len(sys.argv) > 1 else "docs/paper"
    source = os.path.join(paper, "..", "references.md")
    text = open(source, encoding="utf-8").read()
    entries = {}
    for part in re.split(r"^### ", text, flags=re.M)[1:]:
        key, _, body = part.partition("\n")
        entries[key.strip()] = body

    cited = set()
    for path in sorted(glob.glob(os.path.join(paper, "*.md"))):
        if os.path.basename(path) == "references.md":
            continue
        body = open(path, encoding="utf-8").read()

        def link(match):
            key = match.group(1)
            if key not in entries:
                sys.exit(f"{path}: [[{key}]] is not in docs/references.md")
            cited.add(key)
            return f"[[{key}]](references.md#{key.lower()})"

        # Code, such as the arrays of tables of TOML, cites nothing.
        parts = re.split(r"(^```.*?^```|`[^`\n]*`)", body, flags=re.M | re.S)
        linked = "".join(part if part.startswith("`") else CITE.sub(link, part)
                         for part in parts)
        if linked != body:
            open(path, "w", encoding="utf-8").write(linked)

    out = ["# References", "",
           "Each entry was checked against its DOI with "
           "`scripts/paper/verify-references.py`; entries without a DOI "
           "were checked against the page they link to.", ""]
    missing = []
    for key in sorted(cited, key=str.lower):
        body = entries[key]
        if "UNVERIFIED" in body:
            sys.exit(f"{key} is marked UNVERIFIED in docs/references.md")
        # The entry without the notes on its use in MDIR.
        body = re.split(r"^Used for:", body, flags=re.M)[0].strip()
        if "doi.org" not in body:
            missing.append(key)
        out += [f"### {key}", "", body, ""]
    target = os.path.join(paper, "references.md")
    open(target, "w", encoding="utf-8").write("\n".join(out))
    print(f"{len(cited)} entries cited; written to {target}")
    if missing:
        print("without a DOI: " + ", ".join(missing))


if __name__ == "__main__":
    main()
