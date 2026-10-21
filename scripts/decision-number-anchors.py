#!/usr/bin/env python3
"""Keep Markdown heading links valid when a decision label gets its number."""

import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

spec = importlib.util.spec_from_file_location(
    "markdown_links", Path(__file__).with_name("check-markdown-links.py"))
markdown = importlib.util.module_from_spec(spec)
spec.loader.exec_module(markdown)


def tracked_markdown(root):
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "*.md"], cwd=root).split(b"\0")
    return [root / os.fsdecode(name) for name in names if name]


def heading_slugs(source):
    """Return heading anchors in source order, including duplicate suffixes."""
    seen = {}
    previous = None
    result = []
    for _, line in markdown.lines_outside_code(source):
        match = markdown.HEADING.match(line)
        heading = match.group(2) if match else None
        if heading is None and previous is not None and previous.strip() and \
                markdown.SETEXT.match(line) and not markdown.HEADING.match(previous) and \
                not previous.lstrip().startswith(("|", "-", "*", ">")):
            heading = previous.strip()
        if heading is not None:
            base = markdown.slug(heading)
            count = seen.get(base, 0)
            result.append(base if count == 0 else f"{base}-{count}")
            seen[base] = count + 1
        previous = line
    return result


def anchor_changes(source, label, number):
    old = heading_slugs(source)
    new = heading_slugs(source.replace(f"D[{label}]", f"D{number}"))
    return {before: after for before, after in zip(old, new) if before != after}


def update_links(source, path, changes, root):
    outside = {line for line, _ in markdown.lines_outside_code(source)}
    lines = source.splitlines(keepends=True)
    for index, original in enumerate(lines):
        if index + 1 not in outside:
            continue
        masked = markdown.CODE_SPAN.sub(
            lambda match: " " * len(match.group(0)), original)
        matches = list(markdown.INLINE.finditer(masked))
        definition = markdown.DEFINITION.match(masked)
        if definition:
            matches.append(definition)
        edits = []
        for match in matches:
            target = match.group(1)
            wrapped = target.startswith("<") and target.endswith(">")
            value = target[1:-1] if wrapped else target
            if not value or markdown.SCHEME.match(value) or value.startswith("//"):
                continue
            name, separator, fragment = value.partition("#")
            if not separator:
                continue
            if name.startswith("/"):
                resolved = root / unquote(name).lstrip("/")
            else:
                resolved = path.parent / unquote(name) if name else path
            resolved = resolved.resolve()
            replacement = changes.get(resolved, {}).get(unquote(fragment))
            if replacement is None:
                continue
            updated = f"{name}#{replacement}"
            if wrapped:
                updated = f"<{updated}>"
            edits.append((match.start(1), match.end(1), updated))
        for start, end, updated in reversed(edits):
            original = original[:start] + updated + original[end:]
        lines[index] = original
    return "".join(lines)


def assign(root, label, number):
    files = tracked_markdown(root)
    sources = {path: path.read_text(encoding="utf-8") for path in files}
    changes = {path.resolve(): anchor_changes(source, label, number)
               for path, source in sources.items()}
    changed = []
    for path, source in sources.items():
        updated = update_links(source, path, changes, root)
        if updated != source:
            path.write_text(updated, encoding="utf-8")
            changed.append(str(path.relative_to(root)))
    return changed


def check(root):
    # A current heading may legitimately still carry a label in pending mode.
    # Only report decision-shaped fragments that no heading defines.
    stale = re.compile(r"-d[a-z][a-z0-9-]*(?:$|-[0-9]+$)")
    errors = []
    for error in markdown.check(root):
        if error.endswith(": no such heading"):
            target = error.rsplit(": ", 2)[-2]
            fragment = target.partition("#")[2]
            if stale.search(unquote(fragment)):
                errors.append(error)
    return errors


def main(argv):
    root = Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True).strip())
    if argv[1] == "assign":
        for name in assign(root, argv[2], argv[3]):
            print(f"  {name} (links)")
        return 0
    if argv[1] == "check":
        errors = check(root)
        for error in errors:
            print(f"stale decision anchor: {error}", file=sys.stderr)
        return bool(errors)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
