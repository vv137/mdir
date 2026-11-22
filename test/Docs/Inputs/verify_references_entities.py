"""Runs scripts/paper/verify-references.py on one entry, with the Crossref
record given in place of the network: the record's title is escaped as
Crossref returns it for 10.1145/872726.806984 (#884)."""
import importlib.util
import os
import sys
import tempfile

script = sys.argv[1]
spec = importlib.util.spec_from_file_location("verify_references", script)
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)

RECORD = {"message": {
    "title": ["Register allocation &amp; spilling via graph coloring"],
    "published-print": {"date-parts": [[1982, 6]]},
    "author": [{"given": "G. J.", "family": "Chaitin"}],
    "volume": "17",
    "page": "98-101",
}}
verify.fetch = lambda url: RECORD
verify.time.sleep = lambda seconds: None

ENTRY = """### Chaitin1982

G. J. Chaitin, "Register allocation & spilling via graph coloring,"
*SIGPLAN Notices* **17**(6), 98–101 (1982).
[doi:10.1145/872726.806984](https://doi.org/10.1145/872726.806984)
"""
with tempfile.TemporaryDirectory() as directory:
    path = os.path.join(directory, "references.md")
    with open(path, "w", encoding="utf-8") as out:
        out.write(ENTRY)
    sys.argv = [script, path]
    verify.main()
