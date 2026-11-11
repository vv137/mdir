"""Writes an entry of the cache of the skins (format 2, D[skin-budget-record])
for the key that `mdir check --json` reports, whole or with one defect, for
the tests of the reader.

Usage: skin_entry.py CHECK_JSON CACHE_DIR CASE
CASE is `whole` or the name of a defect (see DEFECTS).
"""
import hashlib
import json
import os
import sys

check, directory, case = sys.argv[1:4]
key = json.load(open(check))["run"]["skins"]["cache_key"]
assert key and key.startswith("mdir skins 2\n"), key


def candidate(outer, inner):
    return {"pairlist_distance_angstrom": outer, "pruned_distance_angstrom": inner,
            "predicted_relative": 1.0, "compile_seconds": 10.0, "run_seconds": 3.0,
            "microseconds_per_step": 100.0, "relative_error": 0.01,
            "steps_per_build": 30.0, "steps_per_pruning": 5.0}


entry = {
    "format": 2, "key": key,
    "pairlist_distance_angstrom": 12.0, "pruned_distance_angstrom": 9.6,
    "search": {
        "source": "measured", "reason": "the reference was the fastest",
        "searched": True, "cache": None, "cache_refused": None,
        "not_cached": None, "seconds": 40.0, "budget": 3,
        "budget_basis": {"policy": "always", "share": 0.02, "most": 7,
                         "run_seconds": 3600.0, "candidate_seconds": 13.0,
                         "compile_seconds": 10.0},
        "candidates": [candidate(12.0, 9.6), candidate(14.0, 9.5),
                       candidate(13.0, 9.6)]}}
search = entry["search"]
text = None
DEFECTS = {
    "whole": lambda: None,
    "cut": lambda: None,
    "not-object": lambda: None,
    "no-format": lambda: entry.pop("format"),
    "format-1": lambda: entry.update(format=1),
    "other-key": lambda: entry.update(key=key + "another line\n"),
    "no-outer": lambda: entry.pop("pairlist_distance_angstrom"),
    "no-reaches": lambda: entry.pop("pruned_distance_angstrom"),
    "reach-zero": lambda: entry.update(pairlist_distance_angstrom=0.0),
    "inner-negative": lambda: entry.update(pruned_distance_angstrom=-1.0),
    "no-search": lambda: entry.pop("search"),
    "no-budget": lambda: search.pop("budget"),
    "budget-2": lambda: search.update(budget=2),
    "no-candidates": lambda: search.pop("candidates"),
    "empty-candidates": lambda: search.update(candidates=[]),
    "over-budget": lambda: search["candidates"].append(candidate(15.0, 9.6)),
    "no-reason": lambda: search.pop("reason"),
    "no-seconds": lambda: search.pop("seconds"),
    "candidate-not-object": lambda: search["candidates"].__setitem__(1, 13.5),
    "candidate-no-reach": lambda: search["candidates"][1].pop("pairlist_distance_angstrom"),
    "not-chosen": lambda: entry.update(pruned_distance_angstrom=9.7),
    "no-basis": lambda: search.pop("budget_basis"),
    "basis-no-policy": lambda: search["budget_basis"].pop("policy"),
    "basis-no-share": lambda: search["budget_basis"].pop("share"),
    "candidate-no-pruned": lambda: search["candidates"][1].pop("pruned_distance_angstrom"),
    "candidate-pruned-text": lambda: search["candidates"][1].update(pruned_distance_angstrom="9.5"),
    "candidate-no-predicted": lambda: search["candidates"][1].pop("predicted_relative"),
    "candidate-no-compile": lambda: search["candidates"][1].pop("compile_seconds"),
    "candidate-no-run": lambda: search["candidates"][1].pop("run_seconds"),
    # One list: a candidate's pruned reach of null is 0, which the chosen
    # pruned reach of 0 matches.
    "one-list": lambda: (entry.update(pruned_distance_angstrom=0.0),
                         search["candidates"][0].update(pruned_distance_angstrom=None)),
}
DEFECTS[case]()
text = json.dumps(entry, indent=2)
if case == "cut":
    text = text[:len(text) // 2]
elif case == "not-object":
    text = json.dumps([entry])
os.makedirs(directory, exist_ok=True)
name = hashlib.sha256(key.encode()).hexdigest()[:32] + ".json"
with open(os.path.join(directory, name), "w") as f:
    f.write(text)
