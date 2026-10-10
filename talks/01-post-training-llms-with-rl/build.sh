#!/usr/bin/env bash
# Build the deck "Post-training LLMs with RL".
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
deck="${DECK:-slides}"                      # DECK=tpu-day ./build.sh deck builds tpu-day.md
png="dist/png"; [[ "$deck" != slides ]] && png="dist/png-$deck"
tmp="$(mktemp -d)"
cp -R theme figures "$tmp/"
cp "$deck.md" "$tmp/slides.md"
# parts kept as their own sources are spliced in at <!-- include: <file>.md --> markers
python3 - "$tmp/slides.md" <<'PY'
import re, sys, pathlib
deck = pathlib.Path(sys.argv[1])
text = deck.read_text()
for name in re.findall(r"<!-- include: ([\w.-]+\.md) -->", text):
    part = pathlib.Path(name)
    text = text.replace(f"<!-- include: {name} -->", part.read_text().rstrip() + "\n" if part.exists() else "")
deck.write_text(text)
PY
( cd "$tmp"
  opts=(--theme-set theme/fern.css --html --allow-local-files --no-stdin)
  "$marp_bin" slides.md "${opts[@]}" -o out/slides.html
  "$marp_bin" slides.md "${opts[@]}" --pdf -o out/slides.pdf
  "$marp_bin" slides.md "${opts[@]}" --images png -o out/png/slide.png )
rm -rf dist/figures "$png" && mkdir -p "$png"
cp -R "$tmp/figures" dist/figures
cp "$tmp/out/slides.html" "dist/$deck.html"; cp "$tmp/out/slides.pdf" "dist/$deck.pdf"
cp "$tmp"/out/png/*.png "$png/"
rm -rf "$tmp"
echo "built: dist/$deck.html dist/$deck.pdf $png/"
