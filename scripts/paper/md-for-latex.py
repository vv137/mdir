#!/usr/bin/env python3
"""Prepares a section of the paper for pandoc's LaTeX writer: reads
Markdown on stdin and writes it to stdout with

- citations linked to the headings of the references in the same
  document, (references.md#key) becoming (#key);
- the separator row of each pipe table given dashes in proportion to the
  longest cell of each column, from which pandoc takes the relative widths
  of the columns of a wide table (GitHub ignores them).
"""
import re
import sys


def cells(row):
    # A pipe inside code or math does not split a cell in this paper.
    return [c.strip() for c in row.strip().strip("|").split("|")]


def main():
    lines = sys.stdin.read().replace("](references.md#", "](#").split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if (line.startswith("|") and i + 1 < len(lines)
                and re.fullmatch(r"\|(\s*:?-+:?\s*\|)+\s*", lines[i + 1])):
            table = [line]
            j = i + 2
            while j < len(lines) and lines[j].startswith("|"):
                table.append(lines[j])
                j += 1
            # A mean and its deviation, and a number and its unit, stay on
            # one line.
            table = [re.sub(r"(\d) (s|µs|Å|ps|fs|%)\b", "\\1\u00a0\\2",
                            r.replace(" ± ", "\u00a0±\u00a0")) for r in table]
            line = table[0]
            rows = [cells(r) for r in table]
            n = len(rows[0])
            def column(k):
                return [r[k] if k < len(r) else "" for r in rows]
            longest = [max(len(c) for c in column(k)) for k in range(n)]
            # The longest word of a column, which cannot wrap.
            word = [max((len(w) for c in column(k)
                         for w in re.split(r"[ \t]+", c)),
                        default=3) for k in range(n)]
            # A long cell wraps; give it room, but not all of it, and give
            # every column room for its longest word.
            weights = [max(min(longest[k], 45), word[k] + 2, 4)
                       for k in range(n)]
            out.append(line)
            out.append("|" + "|".join("-" * w for w in weights) + "|")
            out.extend(table[1:])
            i = j
            continue
        out.append(line)
        i += 1
    sys.stdout.write("\n".join(out))


if __name__ == "__main__":
    main()
