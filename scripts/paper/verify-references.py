#!/usr/bin/env python3
"""Checks every entry of docs/references.md that has a DOI against its
Crossref record (DataCite for Zenodo DOIs): the title, the year, the
surname of the first author, the volume, and the first page. Prints one
line for each entry, OK or what differs, and a summary.

    python3 scripts/paper/verify-references.py [docs/references.md]

Needs network access to api.crossref.org and api.datacite.org."""

import html
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request


def fold(text):
    """Lowercase ASCII words of `text`, for comparing titles and names."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"<[^>]+>", " ", text)  # markup in Crossref titles
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def entries(path):
    """(key, text) of each entry."""
    text = open(path, encoding="utf-8").read()
    parts = re.split(r"^### ", text, flags=re.M)[1:]
    for part in parts:
        key, _, body = part.partition("\n")
        yield key.strip(), body


def fetch(url):
    request = urllib.request.Request(
        url, headers={"User-Agent": "mdir-reference-check (mailto:none)"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def record(doi):
    """Title, year, first family name, volume, first page of `doi`."""
    quoted = urllib.parse.quote(doi, safe="")
    if doi.lower().startswith("10.5281/"):
        data = fetch(f"https://api.datacite.org/dois/{quoted}")["data"]
        attributes = data["attributes"]
        creators = attributes.get("creators") or [{}]
        family = creators[0].get("familyName") or creators[0].get("name", "")
        return (attributes["titles"][0]["title"],
                str(attributes.get("publicationYear", "")), family, "", "")
    message = fetch(f"https://api.crossref.org/works/{quoted}")["message"]
    title = (message.get("title") or [""])[0]
    year = ""
    for field in ("published-print", "published-online", "issued",
                  "created"):
        parts = (message.get(field) or {}).get("date-parts") or [[None]]
        if parts[0][0]:
            year = str(parts[0][0])
            break
    authors = message.get("author") or message.get("editor") or [{}]
    family = authors[0].get("family") or authors[0].get("name", "")
    volume = message.get("volume", "")
    page = (message.get("page") or message.get("article-number") or "")
    return title, year, family, volume, page.split("-")[0]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "docs/references.md"
    checked = problems = skipped = 0
    for key, body in entries(path):
        match = re.search(r"\[doi:([^\]]+)\]", body)
        if not match:
            skipped += 1
            print(f"{key}: no DOI")
            continue
        doi = html.unescape(match.group(1).strip())
        flat = " ".join(body.split()).replace("\u2013", "-")
        try:
            title, year, family, volume, page = record(doi)
        except Exception as error:  # network, 404
            problems += 1
            print(f"{key}: {doi}: cannot fetch ({error})")
            continue
        checked += 1
        issues = []
        if fold(title) and fold(title)[:60] not in fold(flat):
            issues.append(f"title {title!r}")
        if year and year not in flat:
            issues.append(f"year {year}")
        if family and fold(family).split()[-1] not in fold(flat):
            issues.append(f"first author {family!r}")
        if volume and volume not in flat:
            issues.append(f"volume {volume}")
        if page and page not in flat:
            issues.append(f"first page {page}")
        if issues:
            problems += 1
            print(f"{key}: {doi}: " + "; ".join(issues))
        else:
            print(f"{key}: OK")
        time.sleep(0.2)
    print(f"\n{checked} DOIs checked, {problems} with differences or errors, "
          f"{skipped} entries without a DOI")


if __name__ == "__main__":
    main()
