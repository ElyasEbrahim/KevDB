#!/usr/bin/env bash
# Build the report PDF: redraw the figures, then typeset with pandoc + XeLaTeX. Needs the .venv at the repository root, pandoc and TeX Live.
set -euo pipefail
cd "$(dirname "$0")"
../../../.venv/bin/python make_figures.py > /dev/null 2>&1

build() {  # build <language> <output file> [extra pandoc arguments...]
  lang=$1; out=$2; shift 2
  # Subscript syntax is off: a tilde in a numeric range is not a subscript
  pandoc "report.$lang.md" -o "$out" --from markdown-subscript --pdf-engine=xelatex --include-in-header=header.tex \
    -V geometry:margin=2.3cm -V fontsize=11pt -V linestretch=1.1 -V colorlinks=true -V linkcolor=black \
    -V mainfont="Times New Roman" -V sansfont="Helvetica Neue" -V monofont="Menlo" "$@"
  echo "$out"
}

build en Jev-databases-report.pdf
