#!/usr/bin/env python3
"""Check the relative links of the Markdown documents of the repository.

Every relative link of README.md, CHANGELOG.md, and docs/**/*.md (except
docs/diagrams/, which is not part of the repository) must name a file or a
directory that exists, and a `#anchor` of a link to a Markdown document must
name a heading of that document, or an HTML anchor (`<a id="...">`) in it.

Anchors follow GitHub's rule for headings: the text of the heading as it is
rendered, in lower case, without the characters other than letters, digits,
underscores, hyphens, and spaces, and with each space made a hyphen; the
second heading of the same anchor takes `-1`, the third `-2`, and so on.

Usage: check-markdown-links.py [ROOT]

Prints each broken link as `file:line: target: reason` and exits with status
1 if there is one, 0 otherwise.
"""

import os
import re
import sys
import unicodedata
from urllib.parse import unquote

FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")
SETEXT = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
HTML_ANCHOR = re.compile(r"""<a\s+[^>]*?(?:id|name)\s*=\s*["']([^"']+)["']""",
                         re.IGNORECASE)
# An inline link or image, `[text](target "title")`; the text may hold one
# level of brackets, as in `[[Key]](references.md#key)`.
INLINE = re.compile(r"!?\[(?:[^\[\]]|\[[^\[\]]*\])*\]\(\s*(<[^>]*>|[^)\s]+)"
                    r"(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\s*\)")
DEFINITION = re.compile(r"^ {0,3}\[[^\]]+\]:\s*(<[^>]*>|\S+)")
CODE_SPAN = re.compile(r"(`+)(.+?)\1")
SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def documents(root):
    """The Markdown documents to check, relative to `root`."""
    found = [name for name in ("README.md", "CHANGELOG.md")
             if os.path.isfile(os.path.join(root, name))]
    docs = os.path.join(root, "docs")
    for directory, subdirectories, files in os.walk(docs):
        relative = os.path.relpath(directory, root)
        if relative == os.path.join("docs", "diagrams"):
            subdirectories[:] = []
            continue
        subdirectories.sort()
        for name in sorted(files):
            if name.endswith(".md"):
                found.append(os.path.join(relative, name))
    return found


def lines_outside_code(text):
    """The lines of `text` with their numbers, those of fenced code blocks
    left out."""
    fence = None
    for number, line in enumerate(text.split("\n"), 1):
        match = FENCE.match(line)
        if fence is None:
            if match:
                fence = match.group(1)
                continue
            yield number, line
        elif match and match.group(1)[0] == fence[0] and \
                len(match.group(1)) >= len(fence) and \
                line.strip() == match.group(1):
            fence = None


def rendered_text(heading):
    """The text of a heading as GitHub renders it, for its anchor."""
    text = heading
    # Images and links keep their text.
    text = re.sub(r"!?\[((?:[^\[\]]|\[[^\[\]]*\])*)\]\([^)]*\)", r"\1", text)
    # Code spans keep their content as written; the rest loses the markers
    # of emphasis.
    parts = []
    last = 0
    for match in CODE_SPAN.finditer(text):
        parts.append(strip_emphasis(text[last:match.start()]))
        parts.append(match.group(2).strip())
        last = match.end()
    parts.append(strip_emphasis(text[last:]))
    text = "".join(parts)
    text = re.sub(r"<[^>]+>", "", text)
    return text


def strip_emphasis(text):
    text = re.sub(r"(\*{1,3})(?=\S)(.+?)(?<=\S)\1", r"\2", text)
    text = re.sub(r"(?<![A-Za-z0-9])(_{1,3})(?=\S)(.+?)(?<=\S)\1(?![A-Za-z0-9])",
                  r"\2", text)
    return text.replace("\\", "")


def slug(heading):
    """GitHub's anchor of a heading, before duplicates are numbered."""
    result = []
    for character in rendered_text(heading).lower():
        category = unicodedata.category(character)
        if character in " -_" or category[0] in "LN" or category == "Mn":
            result.append("-" if character == " " else character)
    return "".join(result)


def anchors(text):
    """Every anchor that a document defines."""
    found = set()
    seen = {}
    previous = None
    for _, line in lines_outside_code(text):
        match = HEADING.match(line)
        heading = match.group(2) if match else None
        if heading is None and previous is not None and previous.strip() and \
                SETEXT.match(line) and not HEADING.match(previous) and \
                not previous.lstrip().startswith(("|", "-", "*", ">")):
            heading = previous.strip()
        if heading is not None:
            base = slug(heading)
            count = seen.get(base, 0)
            found.add(base if count == 0 else "%s-%d" % (base, count))
            seen[base] = count + 1
        for anchor in HTML_ANCHOR.findall(line):
            found.add(anchor)
        previous = line
    return found


def links(text):
    """The targets of the links of a document, with their line numbers."""
    for number, line in lines_outside_code(text):
        line = CODE_SPAN.sub(lambda match: " " * len(match.group(0)), line)
        for match in INLINE.finditer(line):
            yield number, match.group(1)
        match = DEFINITION.match(line)
        if match:
            yield number, match.group(1)


def check(root):
    root = os.path.abspath(root)
    cache = {}
    errors = []

    def anchors_of(path):
        if path not in cache:
            with open(path, encoding="utf-8") as stream:
                cache[path] = anchors(stream.read())
        return cache[path]

    for document in documents(root):
        source = os.path.join(root, document)
        with open(source, encoding="utf-8") as stream:
            text = stream.read()
        for number, target in links(text):
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            if not target or SCHEME.match(target) or target.startswith("//"):
                continue
            path, _, anchor = target.partition("#")
            path = unquote(path)
            if path.startswith("/"):
                resolved = os.path.join(root, path.lstrip("/"))
            elif path:
                resolved = os.path.join(os.path.dirname(source), path)
            else:
                resolved = source
            resolved = os.path.normpath(resolved)
            where = "%s:%d: %s" % (document, number, target)
            if not os.path.exists(resolved):
                errors.append("%s: no such file" % where)
                continue
            if anchor and resolved.endswith(".md") and os.path.isfile(resolved):
                if unquote(anchor) not in anchors_of(resolved):
                    errors.append("%s: no such heading" % where)
    return errors


def main(argv):
    root = argv[1] if len(argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir)
    errors = check(root)
    for error in errors:
        print(error)
    print("%d broken link(s)" % len(errors))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
