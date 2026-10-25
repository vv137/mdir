#!/usr/bin/env python3
"""The index of the decisions in docs/decisions.md, from docs/decisions/.

  decisions-table.py write [ROOT]   write the table between its markers
  decisions-table.py check [ROOT]   exit 1 unless the table is up to date

Each decision is a file of its own (#602): docs/decisions/Dnnn.md for a
numbered one, docs/decisions/<label>.md for one that waits for its number
(D[<label>], docs/workflow.md, Decision numbers). A file holds

    # Dnnn                    (or # D[<label>])

    Recorded YYYY-MM-DD.      (or "Recorded at its merge." before a number)

    **The statement.** The full text of the decision ...

The table has one row a decision, in the order of the numbers and then of
the labels: the link to its file, its statement (the bold sentence that
begins its text), and the date. A PR adds its file and never edits the
table: `decision-number.sh assign` writes it, and `check` checks it.
"""
import os
import re
import sys

BEGIN = "<!-- decisions-table: written by scripts/decisions-table.py from docs/decisions/ -->"
END = "<!-- end decisions-table -->"
HEADING = re.compile(r"^# (D\d+[a-z]?|D\[[a-z0-9-]+\])$")
RECORDED = re.compile(r"^Recorded (\d{4}-\d{2}-\d{2}|at its merge)\.$")


def read(path):
    """The number or label, the date, and the text of a decision file."""
    with open(path, encoding="utf-8") as file:
        lines = file.read().split("\n")
    name = os.path.relpath(path)
    if len(lines) < 5 or not HEADING.match(lines[0]) or lines[1] or \
            not RECORDED.match(lines[2]) or lines[3] or not lines[4].strip():
        raise ValueError(f"{name}: not '# Dnnn', a blank line, 'Recorded "
                         "<date>.', a blank line, and the text")
    number = HEADING.match(lines[0]).group(1)
    stem = os.path.splitext(os.path.basename(path))[0]
    expected = number if not number.startswith("D[") else number[2:-1]
    if stem != expected:
        raise ValueError(f"{name}: its heading is {number}")
    return number, RECORDED.match(lines[2]).group(1), "\n".join(lines[4:]).strip()


def statement(text):
    """The bold sentence that begins the text, its links relative to docs/."""
    m = re.match(r"\*\*(.+?)\*\*", text)
    lead = m.group(1) if m else re.split(r"(?<=\.)\s", text, maxsplit=1)[0]
    lead = lead.replace("\n", " ")
    return re.sub(r"\]\(\.\./", "](", lead)


def key(number):
    m = re.match(r"D(\d+)([a-z]?)$", number)
    return (0, int(m.group(1)), m.group(2), "") if m else (1, 0, "", number)


def table(root):
    directory = os.path.join(root, "docs", "decisions")
    entries = [read(os.path.join(directory, name))
               for name in os.listdir(directory) if name.endswith(".md")]
    entries.sort(key=lambda e: key(e[0]))
    rows = [BEGIN, "| # | Decision | Recorded |", "|---|---|---|"]
    for number, date, text in entries:
        stem = number if not number.startswith("D[") else number[2:-1]
        rows.append(f"| [{number}](decisions/{stem}.md) | {statement(text)} | "
                    f"{date if date != 'at its merge' else 'pending'} |")
    rows.append(END)
    return "\n".join(rows)


def main():
    if len(sys.argv) not in (2, 3) or sys.argv[1] not in ("write", "check"):
        sys.exit(__doc__)
    root = sys.argv[2] if len(sys.argv) == 3 else "."
    path = os.path.join(root, "docs", "decisions.md")
    text = open(path, encoding="utf-8").read()
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        sys.exit(f"{path}: the markers of the table must occur once each")
    try:
        new = table(root)
    except ValueError as error:
        sys.exit(str(error))
    start, end = text.index(BEGIN), text.index(END) + len(END)
    updated = text[:start] + new + text[end:]
    if sys.argv[1] == "write":
        if updated != text:
            open(path, "w", encoding="utf-8").write(updated)
        return
    if updated != text:
        print(f"{path}: the table of decisions is not that of docs/decisions/; "
              "run scripts/decisions-table.py write", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
