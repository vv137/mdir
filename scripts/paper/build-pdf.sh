#!/bin/bash
# Builds the white paper as a PDF: writes each section of docs/paper from
# Markdown to LaTeX with pandoc (OUT/sections/*.tex), copies the master file
# docs/paper/tex/main.tex and the figures, and compiles with tectonic.
#
#   scripts/paper/build-pdf.sh OUT
#
# Needs pandoc (3.x) and tectonic on the path, or in PANDOC and TECTONIC.
# The Markdown is the source; the generated sections are not edited.
set -e
out=$1
pandoc=${PANDOC:-pandoc}
tectonic=${TECTONIC:-tectonic}
paper=$(cd "$(dirname "$0")/../../docs/paper" && pwd)
mkdir -p "$out/sections" "$out/figures"
cp "$paper/tex/main.tex" "$out/"
cp "$paper"/figures/*.png "$out/figures/"

# Citations link to the headings of the references in the same document,
# and wide tables take the widths of their columns from their contents.
convert() {
  python3 "$(dirname "$0")/md-for-latex.py" < "$1" |
    "$pandoc" -f markdown+tex_math_dollars+pipe_tables-implicit_figures \
      -t latex --wrap=preserve --columns=80 \
      --syntax-highlighting=none -o "$2"
}
for f in "$paper"/[0-9]*.md "$paper"/A-*.md "$paper"/B-*.md "$paper"/references.md; do
  convert "$f" "$out/sections/$(basename "$f" .md).tex"
done
# The abstract, from the front page.
sed -n '/^## Abstract/,/^## Contents/p' "$paper/README.md" | sed '1d;$d' |
  "$pandoc" -f markdown+tex_math_dollars -t latex -o "$out/sections/abstract.tex"

# Record when this PDF is compiled, independently of the paper's edition date.
printf '\\newcommand{\\paperbuildtime}{%s}\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" > "$out/build-info.tex"
cd "$out" && "$tectonic" -X compile --keep-logs main.tex
echo "$out/main.pdf"
