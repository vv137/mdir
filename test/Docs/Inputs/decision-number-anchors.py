"""Exercise decision numbering in a small tracked Markdown tree."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

repository = Path(sys.argv[1]).resolve()
foo = "D" + "[foo]"
foo_bar = "D" + "[foo-bar]"
with tempfile.TemporaryDirectory(prefix="mdir-decision-links-") as temporary:
    root = Path(temporary)
    (root / "scripts").mkdir()
    (root / "docs").mkdir()
    for script in ("decision-number.sh", "decision-number-anchors.py",
                   "check-markdown-links.py"):
        shutil.copy2(repository / "scripts" / script, root / "scripts" / script)
    (root / "docs" / "decisions.md").write_text(
        f"| D315 | Existing |\n| {foo} | Pending |\n"
        f"| {foo_bar} | Another pending decision |\n", encoding="utf-8")
    (root / "docs" / "topic.md").write_text(
        f"## GPU modules ({foo})\n\n## GPU modules ({foo})\n"
        f"\n## GPU modules ({foo_bar})\n", encoding="utf-8")
    (root / "README.md").write_text(
        "[one](docs/topic.md#gpu-modules-dfoo)\n"
        "[two](docs/topic.md#gpu-modules-dfoo-1)\n"
        "[other](docs/topic.md#gpu-modules-dfoo-bar)\n"
        "[reference][section]\n\n[section]: <docs/topic.md#gpu-modules-dfoo>\n"
        "`[code](docs/topic.md#gpu-modules-dfoo)`\n"
        "```md\n[fence](docs/topic.md#gpu-modules-dfoo)\n```\n",
        encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "README.md", "docs", "scripts"],
                   cwd=root, check=True)

    def run(*args):
        return subprocess.run(["bash", "scripts/decision-number.sh", *args],
                              cwd=root, text=True, capture_output=True)

    assert run("check", "--pending").returncode == 0
    numbered = run("assign", "foo")
    assert numbered.returncode == 0, numbered.stderr
    links = (root / "README.md").read_text(encoding="utf-8")
    assert "#gpu-modules-d316)" in links
    assert "#gpu-modules-d316-1)" in links
    assert "#gpu-modules-dfoo-bar)" in links
    assert "[section]: <docs/topic.md#gpu-modules-d316>" in links
    assert "`[code](docs/topic.md#gpu-modules-dfoo)`" in links
    assert "[fence](docs/topic.md#gpu-modules-dfoo)" in links
    assert run("check").returncode == 1  # The second label remains pending.
    assert run("check", "--pending").returncode == 0
    (root / "README.md").write_text(
        links + "[stale](docs/topic.md#gpu-modules-dfoo)\n", encoding="utf-8")
    checked = run("check", "--pending")
    assert checked.returncode == 1
    assert "stale decision anchor" in checked.stderr
    print("decision heading links: ok")
