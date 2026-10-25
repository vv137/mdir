"""Run scripts/release/check-path.sh (#609) on a small repository made of the
release scripts and a tree that holds what they edit: once with a test that
pins the version of the tree in a FileCheck pattern, which it reports, and
once with that pattern matched to any version, which passes."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

repository = Path(sys.argv[1]).resolve()
BEGIN = ("<!-- decisions-table: written by scripts/decisions-table.py "
         "from docs/decisions/ -->")
END = "<!-- end decisions-table -->"
scripts = ["check-markdown-links.py", "decision-number.sh",
           "decision-number-anchors.py", "decisions-table.py",
           "release/bump.sh", "release/changelog.py", "release/check-path.sh",
           "release/version-pins.txt"]
with tempfile.TemporaryDirectory(prefix="mdir-release-path-") as temporary:
    root = Path(temporary)
    for script in scripts:
        (root / "scripts" / script).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repository / "scripts" / script, root / "scripts" / script)
    (root / "scripts" / "release" / "version-pins.txt").write_text(
        "# Nothing is pinned here.\n", encoding="utf-8")
    files = {
        "CMakeLists.txt": "project(mdir VERSION 1.4.0 LANGUAGES CXX C)\n",
        "examples/ala3/run.py": '# dependencies = ["mdir[cuda]==1.4.0"]\n',
        "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n### Fixed\n\n"
                        "- A fix, see [the topic](docs/topic.md#notes).\n\n"
                        "## [1.4.0] - 2026-10-01\n\n### Added\n\n- Before.\n\n"
                        "[Unreleased]: https://github.com/vv137/mdir/compare/v1.4.0...HEAD\n"
                        "[1.4.0]: https://github.com/vv137/mdir/releases/tag/v1.4.0\n",
        "changelog.d/fix.md": "### Added\n\n- A fragment, in [the docs](docs/topic.md).\n",
        "docs/topic.md": "# Topic\n\n## Notes\n",
        "docs/decisions.md": f"# Decisions\n\n{BEGIN}\n{END}\n",
        "docs/decisions/D1.md": "# D1\n\nRecorded 2026-10-01.\n\n**One.**\n",
        "test/Driver/version.test": "# RUN: mdir --version | FileCheck %s\n"
                                    "# A comment that names 1.4.0 is not read.\n"
                                    "# CHECK: MDIR 1.4.0\n",
        "lib/Version.cpp": '// 1.4.0 in a comment.\nconst char *v = "1.4.0";\n',
    }
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")
    subprocess.run(["python3", "scripts/decisions-table.py", "write"], cwd=root, check=True)

    def commit():
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@invalid",
                        "commit", "-q", "-m", "tree"], cwd=root, check=True)

    def run():
        return subprocess.run(["bash", "scripts/release/check-path.sh"], cwd=root,
                              text=True, capture_output=True,
                              env={"PATH": "/usr/bin:/bin", "TMPDIR": temporary})

    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    commit()
    first = run()
    print(first.stdout + first.stderr, end="")
    (root / "test/Driver/version.test").write_text(
        "# RUN: mdir --version | FileCheck %s\n"
        "# CHECK: MDIR {{[0-9]+\\.[0-9]+\\.[0-9]+}}\n", encoding="utf-8")
    (root / "lib/Version.cpp").write_text("const char *v = MDIR_VERSION;\n",
                                          encoding="utf-8")
    commit()
    second = run()
    print("-- after the fix --")
    print(second.stdout + second.stderr, end="")
    # The tree itself is never changed.
    assert (root / "changelog.d/fix.md").exists()
    assert "1.4.0 LANGUAGES" in (root / "CMakeLists.txt").read_text()
    print(f"status {first.returncode} then {second.returncode}")
