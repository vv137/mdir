"""Exercise decision numbering in a small tracked Markdown tree: the files of
the decisions (docs/decisions/, #602), their table, and heading links."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

repository = Path(sys.argv[1]).resolve()
foo = "D" + "[foo]"
foo_bar = "D" + "[foo-bar]"
BEGIN = ("<!-- decisions-table: written by scripts/decisions-table.py "
         "from docs/decisions/ -->")
END = "<!-- end decisions-table -->"
with tempfile.TemporaryDirectory(prefix="mdir-decision-links-") as temporary:
    root = Path(temporary)
    (root / "scripts").mkdir()
    (root / "docs" / "decisions").mkdir(parents=True)
    for script in ("decision-number.sh", "decision-number-anchors.py",
                   "decisions-table.py", "check-markdown-links.py"):
        shutil.copy2(repository / "scripts" / script, root / "scripts" / script)
    (root / "docs" / "decisions.md").write_text(
        f"# Decisions\n\n{BEGIN}\n{END}\n", encoding="utf-8")
    (root / "docs" / "decisions" / "D315.md").write_text(
        "# D315\n\nRecorded 2026-10-20.\n\n**Existing.** See "
        "[the topic](../topic.md).\n", encoding="utf-8")
    (root / "docs" / "decisions" / "foo.md").write_text(
        f"# {foo}\n\nRecorded at its merge.\n\n**Pending.** After D315.\n",
        encoding="utf-8")
    (root / "docs" / "decisions" / "foo-bar.md").write_text(
        f"# {foo_bar}\n\nRecorded at its merge.\n\n**Another pending "
        "decision.**\n", encoding="utf-8")
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

    def run(*args):
        return subprocess.run(["bash", "scripts/decision-number.sh", *args],
                              cwd=root, text=True, capture_output=True)

    def table(*args):
        return subprocess.run(["python3", "scripts/decisions-table.py", *args],
                              cwd=root, text=True, capture_output=True)

    # The table is written from the files, and check sees a stale one.
    assert table("check").returncode == 1
    assert table("write").returncode == 0, table("write").stderr
    log = (root / "docs" / "decisions.md").read_text(encoding="utf-8")
    assert "| [D315](decisions/D315.md) | Existing. | 2026-10-20 |" in log, log
    assert f"| [{foo}](decisions/foo.md) | Pending. | pending |" in log, log
    subprocess.run(["git", "add", "README.md", "docs", "scripts"],
                   cwd=root, check=True)

    assert run("check", "--pending").returncode == 0, run("check", "--pending").stderr
    numbered = run("assign", "foo")
    assert numbered.returncode == 0, numbered.stderr
    assert not (root / "docs" / "decisions" / "foo.md").exists()
    text = (root / "docs" / "decisions" / "D316.md").read_text(encoding="utf-8")
    assert text.startswith("# D316\n\nRecorded 20"), text
    assert "Recorded at its merge" not in text
    log = (root / "docs" / "decisions.md").read_text(encoding="utf-8")
    assert "| [D316](decisions/D316.md) | Pending. |" in log, log
    assert table("check").returncode == 0
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
    (root / "README.md").write_text(links, encoding="utf-8")
    # A label without its file, and a row of the earlier form, are refused.
    (root / "docs" / "decisions" / "foo-bar.md").unlink()
    checked = run("check", "--pending")
    assert checked.returncode == 1
    assert "without a file in docs/decisions/" in checked.stderr, checked.stderr
    subprocess.run(["git", "checkout", "-q", "docs/decisions/foo-bar.md"],
                   cwd=root, check=True)
    (root / "docs" / "decisions.md").write_text(
        log + "| D317 | A row |\n", encoding="utf-8")
    checked = run("check", "--pending")
    assert checked.returncode == 1
    assert "a decision is a file" in checked.stderr, checked.stderr
    print("decision heading links: ok")
