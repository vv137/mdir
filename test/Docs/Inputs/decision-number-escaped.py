"""Number a decision whose label a test writes escaped (#1010), in a small
tracked tree with copies of the scripts."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

repository = Path(sys.argv[1]).resolve()


def text(template):
    """Q before a bracket stands for D, so that this file holds no label."""
    return template.replace("Q", "D")


BEGIN = ("<!-- decisions-table: written by scripts/decisions-table.py "
         "from docs/decisions/ -->")
END = "<!-- end decisions-table -->"
PY = r'''import re
LOG = "MDIR: the grid (D444)"
assert re.search(r"\(Q\[esc\]\)", LOG)
assert re.search("Q\\[esc\\]", LOG)
assert re.search(r"\bQ\[esc\]", LOG)
assert re.search(r"Q[[]esc[]]", LOG)
assert re.search(r"Q\[esc]", LOG)
assert re.search(r"\(Q\[other\]\)", LOG)
'''
PY_AFTER = r'''import re
LOG = "MDIR: the grid (D444)"
assert re.search(r"\(D316\)", LOG)
assert re.search("D316", LOG)
assert re.search(r"\bD316", LOG)
assert re.search(r"D316", LOG)
assert re.search(r"D316", LOG)
assert re.search(r"\(Q\[other\]\)", LOG)
'''
TEST = r'''# RUN: mdir run x.toml | FileCheck %s
# CHECK: the grid {{\(Q\[esc\]\)}}
# CHECK: the grid (Q{{\[}}esc{{\]}})
# CHECK: the grid (Q{{[[]}}esc{{[]]}})
# CHECK: the grid (Q[esc])
# CHECK: the grid {{.*}}Q{{\[}}other{{\]}}
'''
TEST_AFTER = r'''# RUN: mdir run x.toml | FileCheck %s
# CHECK: the grid {{\(D316\)}}
# CHECK: the grid (D316)
# CHECK: the grid (D316)
# CHECK: the grid (D316)
# CHECK: the grid {{.*}}Q{{\[}}other{{\]}}
'''

with tempfile.TemporaryDirectory(prefix="mdir-decision-escaped-") as temporary:
    root = Path(temporary)
    (root / "scripts").mkdir()
    (root / "docs" / "decisions").mkdir(parents=True)
    (root / "test").mkdir()
    for script in ("decision-number.sh", "decision-number-anchors.py",
                   "decisions-table.py", "check-markdown-links.py"):
        shutil.copy2(repository / "scripts" / script, root / "scripts" / script)
    (root / "docs" / "decisions.md").write_text(
        f"# Decisions\n\n{BEGIN}\n{END}\n", encoding="utf-8")
    (root / "docs" / "decisions" / "D315.md").write_text(
        "# D315\n\nRecorded 2026-10-20.\n\n**Existing.**\n", encoding="utf-8")
    for label in ("esc", "other", "net"):
        (root / "docs" / "decisions" / f"{label}.md").write_text(
            text(f"# Q[{label}]\n\nRecorded at its merge.\n\n**Pending.**\n"),
            encoding="utf-8")
    py = root / "test" / "grid.py"
    lit = root / "test" / "grid.test"
    py.write_text(text(PY), encoding="utf-8")
    lit.write_text(text(TEST), encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)

    def run(*args):
        return subprocess.run(["bash", "scripts/decision-number.sh", *args],
                              cwd=root, text=True, capture_output=True)

    def table_write():
        subprocess.run(["python3", "scripts/decisions-table.py", "write"],
                       cwd=root, check=True)

    def add():
        subprocess.run(["git", "add", "docs", "scripts", "test"],
                       cwd=root, check=True)

    table_write()
    add()
    checked = run("check", "--pending")
    assert checked.returncode == 0, checked.stderr
    for label in ("esc", "other", "net"):
        assert text(f"waiting for a number: Q[{label}]") in checked.stdout, \
            checked.stdout

    # assign rewrites every escaped form of the label in both files, and
    # leaves the other label alone.
    numbered = run("assign", "esc")
    assert numbered.returncode == 0, numbered.stderr
    assert py.read_text(encoding="utf-8") == text(PY_AFTER), py.read_text()
    assert lit.read_text(encoding="utf-8") == text(TEST_AFTER), lit.read_text()
    assert "test/grid.py" in numbered.stdout, numbered.stdout
    assert "test/grid.test" in numbered.stdout, numbered.stdout
    add()
    checked = run("check", "--pending")
    assert checked.returncode == 0, checked.stderr

    # A form of the numbered label left in either file fails check, which
    # names its file and line; the plain form too.
    for path, line, number in (
            (py, r'assert re.search(r"\(Q\[esc\]\)", LOG)', 9),
            (py, 'assert re.search("Q\\\\[esc\\\\]", LOG)', 9),
            (lit, "# CHECK: the grid (Q{{\\[}}esc{{\\]}})", 7),
            (lit, "# CHECK: the grid (Q{{[[]}}esc{{[]]}})", 7),
            (lit, "# CHECK: the grid Q[esc]", 7)):
        before = path.read_text(encoding="utf-8")
        path.write_text(before + text(line) + "\n", encoding="utf-8")
        add()
        checked = run("check", "--pending")
        assert checked.returncode == 1, (line, checked.stdout)
        assert "without a file in docs/decisions/" in checked.stderr, \
            checked.stderr
        where = f"test/{path.name}:{number}:"
        assert where in checked.stderr, (where, checked.stderr)
        strict = run("check")
        assert strict.returncode == 1
        assert where in strict.stderr, (where, strict.stderr)
        path.write_text(before, encoding="utf-8")
        add()

    # A label that never had a file fails as well, in an escaped form.
    before = py.read_text(encoding="utf-8")
    py.write_text(before + text(r'assert re.search(r"Q\[nofile\]", LOG)')
                  + "\n", encoding="utf-8")
    add()
    checked = run("check", "--pending")
    assert checked.returncode == 1
    assert "test/grid.py:9:" in checked.stderr, checked.stderr
    py.write_text(before, encoding="utf-8")
    add()

    # A form that assign does not rewrite is refused by assign, named.
    lit.write_text(lit.read_text(encoding="utf-8")
                   + text("# CHECK: {{Q[\\[]net[\\]]}}\n"), encoding="utf-8")
    add()
    refused = run("assign", "net")
    assert refused.returncode == 1, refused.stdout
    assert "test/grid.test:7:" in refused.stderr, refused.stderr
    assert "a form that assign does not rewrite" in refused.stderr, \
        refused.stderr
    print("decision labels in escaped forms: ok")
