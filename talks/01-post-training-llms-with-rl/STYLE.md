# How this deck is made: tools, style and rules

This deck follows the house style of the FME office-hours deck: black text, greys for structure,
and one accent colour, which here is fern green.

## Build

| Job | Command |
|---|---|
| One figure | `./build.sh figures <name>`: `tikz/<name>.tex` → `figures/<name>.svg` and `dist/fig-png/<name>.png` |
| All figures + deck | `./build.sh`: renders HTML, PDF and PNG previews to `dist/` |
| Layout check | `node tools/check-layout.mjs dist/slides.html`. It must pass before sharing. |
| Slide code | `python code/pt_snippets.py`. Every snippet on a slide is copied from here and tested. |
| Slide numbers | `python code/check_numbers.py`. Every number on a slide or figure is computed and checked. |

- **First-time setup:** `npm install` (Marp CLI is pinned in `package.json`) and `.venv` with
  `jax[cpu]`.
- **How the build works:** figures compile with XeLaTeX in two passes. The deck renders in a temp
  copy with `--no-stdin`, because headless Chrome hangs when writing into `~/Desktop`.

## Visual style

- **Fonts:**
  - Urbanist for headings.
  - Avenir Next for body text and figures.
  - Menlo for code.
- **Palette:**
  - ink `#111111`
  - greys `#4A4F55` / `#6B7178` / `#8A9097` / `#D5D9DD` / `#F4F5F6`
  - fern `#4F7942` / `#2F4A28`, with tint `#EAF1E6`
- **Every slide:**
  - White background, with no gradients and no dark slides.
  - Header: "Prabakaran Chandran | pracha.me".
  - Footer: the citation for the slide, public sources only.
- **Slide classes:**

  | Class | Use |
  |---|---|
  | `title` | the opening slide |
  | `topics` | the outline |
  | `divider` | opens each part |
  | `figure` / `tall` | one figure plus a one-paragraph reading |
  | `figtext` | a figure plus up to two bullets |
  | `concept` | Definition · Intuition · Why · Further reading |
  | `dense small` | tables and code |

- **Titles state the point** (e.g. "GRPO removes the critic"), not the format.
- **No scaffolding labels**: no "under the hood", "so far" or "start here".

## Figure protocol (every TikZ figure)

All shared styles live in `tikz/pt-style.tex`.

1. **One idea per figure, one accent.** Fern marks the focus (usually the model being trained).
   Grey fill marks frozen or external parts.
2. **Diagrams:**
   - Arrows start and end on node anchors and are orthogonal (`-|`, `|-`) or straight.
   - Curves only for real loops.
   - Branches go through a junction dot.
   - Arrows never cross boxes or text.
   - Parallel boxes keep a visible gap.
3. **Explanations go in the numbered notes column**, never as floating text:
   - markers `\mk` / `\mka` on the drawing;
   - `\notesat` + `\note` / `\notea` on the right;
   - a `\symbols` strip under the notes defining every letter.
4. **Annotated equations use `\ann` / `\anna` only:**
   - The formula is one node at the origin (`anchor=base west`), with terms wrapped in `\tm`.
   - Labels sit in fixed rows above or below (tier 1 or 2), centred on their term.
   - Each label joins its term by a straight vertical leader with a dot. Side labels (`l` / `r`)
     are for the outermost terms only.
   - Tier-1 labels are at most two lines. A tier-2 leader must pass between tier-1 labels.
5. **Glyphs:**
   - linear projection = trapezoid (`expand` / `contract`), dimensions inside the glyph;
   - activation = `silubox`;
   - residual stream = thick grey bar;
   - elementwise gate = ⊗.
6. **Never type a number**; compute it (`code/check_numbers.py`). Label illustrative values
   "illustrative".
7. **Review the PNG after every change**: at the figure itself, and at slide size in
   `dist/png/`.

## Notation (one meaning per symbol, on every slide)

| Symbol | Meaning |
|---|---|
| $x$ | prompt |
| $y=(y_1,\dots,y_T)$ | response |
| $s_t=(x,y_{<t})$ | state |
| $a_t=y_t$ | action |
| $\pi_\theta$ | policy being trained |
| $\pi_{\text{old}}$ | sampling snapshot |
| $\pi_{\text{ref}}$ | frozen starting model |
| $r$, $r_\phi$ | reward, reward model |
| $V_\psi$ | critic |
| $\hat A_t$ | advantage |
| $\rho_t$ | probability ratio (PPO's $r_t$, renamed) |
| $\varepsilon$, $\beta$ | clip range, KL weight |
| $G$ | group size |
| $\pi_T$ | teacher |
| $y_w$, $y_l$ | preferred, rejected response |
| $\pi^{*}$, $Z(x)$ | optimal policy for a reward, its normalizer (DPO) |
| $\hat r_\theta$ | DPO's implicit reward, $\beta\log\pi_\theta/\pi_{\text{ref}}$ |
| $y^\star$ | reference solution |

## Grounding

Every claim cites a primary source: a paper with its section, equation or table, a model card or
config, or a lab's own post. Section numbers appear only after they have been checked in the
source. Uncertain or contested results are labelled as such, not smoothed over.
