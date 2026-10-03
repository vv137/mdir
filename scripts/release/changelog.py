#!/usr/bin/env python3
"""Release edits of CHANGELOG.md (Keep a Changelog).

  changelog.py release VERSION DATE   move [Unreleased] under [VERSION] - DATE
  changelog.py notes VERSION          print the section of VERSION
  changelog.py check VERSION          exit 1 unless VERSION has a dated section

`release` merges the entries of [Unreleased] into an existing
"[VERSION] - unreleased" section subsection by subsection (Added, Changed,
Fixed, ...), or makes a new section above the previous version, and leaves
[Unreleased] empty. The compare links at the end are updated.
"""
import re
import sys

PATH = "CHANGELOG.md"
ORDER = ["Added", "Changed", "Deprecated", "Removed", "Fixed", "Security",
         "Validated", "Performance", "Known limitations"]


def split_sections(text):
    """Head, then (title, body) for each '## ' section, then the link tail."""
    parts = re.split(r"(?m)^(## .*)$", text)
    head, sections = parts[0], []
    for i in range(1, len(parts), 2):
        sections.append([parts[i], parts[i + 1]])
    tail = ""
    if sections:
        m = re.search(r"(?m)^\[[^\]]+\]: ", sections[-1][1])
        if m:
            tail = sections[-1][1][m.start():]
            sections[-1][1] = sections[-1][1][:m.start()]
    return head, sections, tail


def subsections(body):
    """Intro text, then an ordered dict of '### ' subsection -> entries."""
    parts = re.split(r"(?m)^### (.*)$", body)
    intro, subs = parts[0], {}
    for i in range(1, len(parts), 2):
        subs[parts[i].strip()] = parts[i + 1].strip("\n")
    return intro, subs


def join_subsections(intro, subs):
    out = intro.rstrip("\n") + "\n\n" if intro.strip() else "\n"
    names = [n for n in ORDER if n in subs] + [n for n in subs if n not in ORDER]
    for name in names:
        if subs[name].strip():
            out += f"### {name}\n\n{subs[name].strip()}\n\n"
    return out


def release(version, date):
    text = open(PATH).read()
    head, sections, tail = split_sections(text)
    titles = [t for t, _ in sections]
    if not titles or titles[0] != "## [Unreleased]":
        sys.exit("CHANGELOG.md: the first section must be [Unreleased]")
    if any(t.startswith(f"## [{version}] - ") and not t.endswith("unreleased")
           for t in titles):
        sys.exit(f"CHANGELOG.md: {version} is already released")
    _, new = subsections(sections[0][1])
    if not any(v.strip() for v in new.values()):
        sys.exit("CHANGELOG.md: [Unreleased] is empty")
    target = next((i for i, t in enumerate(titles)
                   if t == f"## [{version}] - unreleased"), None)
    if target is None:
        sections.insert(1, [f"## [{version}] - {date}", "\n"])
        target = 1
    else:
        sections[target][0] = f"## [{version}] - {date}"
    intro, old = subsections(sections[target][1])
    for name, entries in new.items():
        if entries.strip():
            old[name] = (entries.strip() + "\n" + old.get(name, "")).strip()
    sections[target][1] = join_subsections(intro, old)
    sections[0][1] = "\n\n"
    tail = re.sub(r"(?m)^\[Unreleased\]: .*$",
                  f"[Unreleased]: https://github.com/vv137/mdir/compare/"
                  f"v{version}...HEAD", tail)
    if not re.search(rf"(?m)^\[{re.escape(version)}\]: ", tail):
        tail = tail.rstrip("\n") + (f"\n[{version}]: https://github.com/vv137/"
                                    f"mdir/releases/tag/v{version}\n")
    out = head + "".join(t + "\n" + b if not b.startswith("\n") else t + b
                         for t, b in sections) + tail
    open(PATH, "w").write(re.sub(r"\n{3,}", "\n\n", out))


def section(version):
    _, sections, _ = split_sections(open(PATH).read())
    for title, body in sections:
        if title.startswith(f"## [{version}] - "):
            return title, body
    return None, None


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    command, version = sys.argv[1], sys.argv[2]
    if command == "release" and len(sys.argv) == 4:
        release(version, sys.argv[3])
    elif command == "notes":
        title, body = section(version)
        if title is None:
            sys.exit(f"CHANGELOG.md: no section for {version}")
        print(body.strip())
    elif command == "check":
        title, _ = section(version)
        if title is None or title.endswith("unreleased"):
            sys.exit(f"CHANGELOG.md: {version} has no dated section")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
