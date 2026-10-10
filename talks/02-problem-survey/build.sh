#!/usr/bin/env bash
# Build the deck "The Problem Landscape at CVPR and ACL".
#   ./build.sh figures   TikZ -> SVG (+ PNG previews) only; optional: ./build.sh figures name
#   ./build.sh           figures, then deck -> dist/ (HTML, PDF, PNG previews)
#   ./build.sh deck      deck only, reusing figures/*.svg
# The deck renders in a temp copy: headless Chrome can hang when asked to write into ~/Desktop,
# and marp waits on stdin without a terminal unless --no-stdin is given.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"
mkdir -p dist/tikz figures dist/fig-png

build_fig() {
  local name="$1"
  for pass in 1 2; do   # two passes: annotated equations need remembered node positions
    (cd tikz && xelatex -interaction=nonstopmode -halt-on-error -output-directory=../dist/tikz "$name.tex" >/dev/null) \
      || { echo "LaTeX failed: tikz/$name.tex (see dist/tikz/$name.log)"; return 1; }
  done
  pdftocairo -svg "dist/tikz/$name.pdf" "figures/$name.svg"
  pdftocairo -png -r 200 -singlefile "dist/tikz/$name.pdf" "dist/fig-png/$name"
  echo "figure: figures/$name.svg"
}

if [[ "${1:-}" == "figures" && -n "${2:-}" ]]; then build_fig "$2"; exit 0; fi
if [[ "${1:-}" != "deck" ]]; then   # ./build.sh deck: reuse the existing figures
for tex in tikz/*.tex; do
  name=$(basename "$tex" .tex)
  [[ "$name" == pt-style ]] && continue
  build_fig "$name"
done
fi
[[ "${1:-}" == "figures" ]] && exit 0

marp_bin="$here/node_modules/.bin/marp"     # pinned in package.json (npm install)
tmp="$(mktemp -d)"
cp -R slides.md theme figures "$tmp/"
( cd "$tmp"
  opts=(--theme-set theme/fern.css --html --allow-local-files --no-stdin)
  "$marp_bin" slides.md "${opts[@]}" -o out/slides.html
  "$marp_bin" slides.md "${opts[@]}" --pdf -o out/slides.pdf
  "$marp_bin" slides.md "${opts[@]}" --images png -o out/png/slide.png )
rm -rf dist/figures dist/png && mkdir -p dist/png
cp -R "$tmp/figures" dist/figures
cp "$tmp/out/slides.html" "$tmp/out/slides.pdf" dist/
cp "$tmp"/out/png/*.png dist/png/
rm -rf "$tmp"
echo "built: dist/slides.html dist/slides.pdf dist/png/"
