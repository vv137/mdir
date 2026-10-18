"""Lists the tests that write outputs at fixed %t paths without removing them
first (#423).

`mdir run` and the Python interface keep an output that exists as a
numbered backup, '#name.N#', and refuse to write once its directory holds 99
of them. A test that writes at a fixed %t path without removing it first
therefore leaves one backup more in the build tree at every run of the
suite, and fails after 99 runs. A test is listed when its RUN lines run
`mdir run` or a Python interpreter (which may run either) and use a path
under %t whose root (`%t`, `%t-mixed`, `%t.dcd`, ...) no `rm` removes on or
before the line where the root first appears. The usual form is
`rm -rf %t && mkdir -p %t` on the first RUN line, with every path under it.

Usage: check_test_outputs.py DIRECTORY...; prints one line for each test
listed, and nothing when there is none.
"""

import fnmatch
import pathlib
import re
import sys

RUNS = re.compile(r"\bmdir\s+run\b|%[a-z_]*python\b")
# %t, not a longer substitution that starts with it (%torch_python).
TEMPORARY = re.compile(r"%t(?![A-Za-z_{])[^\s'\"|;&()<>,=]*")
REMOVE = re.compile(r"(?:^|[;&|]\s*|\s)rm\s+([^;&|]*)")


def run_lines(text):
    lines, line = [], ""
    for source in text.splitlines():
        match = re.search(r"\bRUN:(.*)$", source)
        if not match:
            continue
        part = match.group(1).rstrip()
        if part.endswith("\\"):
            line += part[:-1] + " "
            continue
        lines.append(line + part)
        line = ""
    if line:
        lines.append(line)
    return lines


def root(path):
    slash = path.find("/")
    return path if slash < 0 else path[:slash]


def missing(text):
    lines = run_lines(text)
    if not any(RUNS.search(line) for line in lines):
        return []
    first = {}
    removed = []
    for number, line in enumerate(lines):
        for path in TEMPORARY.findall(line):
            first.setdefault(root(path), number)
        for match in REMOVE.finditer(line):
            removed += [(argument, number) for argument in match.group(1).split()
                        if not argument.startswith("-")]
    return sorted(name for name, number in first.items()
                  if not any(fnmatch.fnmatchcase(name, argument) and at <= number
                             for argument, at in removed))


def main():
    for directory in map(pathlib.Path, sys.argv[1:]):
        for test in sorted(directory.rglob("*.test")):
            if "Inputs" in test.relative_to(directory).parts:
                continue
            names = missing(test.read_text(errors="replace"))
            if names:
                print(f"{test.relative_to(directory)}: {' '.join(names)}")


if __name__ == "__main__":
    main()
