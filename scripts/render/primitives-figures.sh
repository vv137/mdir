#!/bin/sh
# Regenerates the SVGs of docs/figures/primitives from their .dot sources
# (docs/primitives-map.md). Requires Graphviz's `dot`; DOT names another
# binary.
set -eu
dir=$(cd "$(dirname "$0")/../../docs/figures/primitives" && pwd)
dot=${DOT:-dot}
if ! command -v "$dot" >/dev/null 2>&1; then
  echo "primitives-figures.sh: Graphviz's dot not found; set DOT" >&2
  exit 1
fi
for source in "$dir"/*.dot; do
  "$dot" -Tsvg "$source" -o "${source%.dot}.svg"
  echo "wrote ${source%.dot}.svg"
done
