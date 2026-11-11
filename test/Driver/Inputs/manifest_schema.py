"""A validator of the small JSON Schema vocabulary that
docs/run-manifest.schema.json uses, without a Python package in the build
environment (manifest.py). Run as a script, it validates each line of the
manifests given against the schema.

Usage: manifest_schema.py SCHEMA MANIFEST...
"""
import json
import re
import sys

# Validate the small JSON Schema vocabulary used by the public schema without
# requiring a Python package in the build environment.

def validate(value, rule):
    for keyword in ("oneOf", "anyOf"):
        if keyword in rule:
            matches = 0
            for choice in rule[keyword]:
                try:
                    validate(value, choice)
                    matches += 1
                except AssertionError:
                    pass
            assert matches == 1 if keyword == "oneOf" else matches >= 1, (value, rule)
    if "type" in rule:
        types = rule["type"] if isinstance(rule["type"], list) else [rule["type"]]
        kind = {dict: "object", list: "array", str: "string", bool: "boolean",
                int: "integer", float: "number", type(None): "null"}[type(value)]
        assert kind in types or (kind == "integer" and "number" in types), (value, rule)
    if "const" in rule:
        assert value == rule["const"] and type(value) is type(rule["const"])
    if "enum" in rule:
        assert value in rule["enum"]
    if "minimum" in rule:
        assert value >= rule["minimum"]
    if "pattern" in rule:
        assert re.search(rule["pattern"], value)
    if isinstance(value, dict):
        assert set(rule.get("required", ())) <= value.keys()
        properties = rule.get("properties", {})
        for key, item in value.items():
            if key in properties:
                validate(item, properties[key])
            elif rule.get("additionalProperties") is False:
                assert False, key
            elif isinstance(rule.get("additionalProperties"), dict):
                validate(item, rule["additionalProperties"])
    if isinstance(value, list):
        assert rule.get("minItems", 0) <= len(value) <= rule.get("maxItems", len(value))
        for item in value:
            if "items" in rule:
                validate(item, rule["items"])


if __name__ == "__main__":
    schema = json.load(open(sys.argv[1]))
    for path in sys.argv[2:]:
        for line in open(path, encoding="utf-8"):
            validate(json.loads(line), schema)
        print("valid", path.rsplit("/", 1)[-1])
