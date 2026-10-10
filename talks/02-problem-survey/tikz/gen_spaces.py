"""Problem-space figures from tikz/data/spaces_{cvpr,acl}.csv.

  tax-*.tex    taxonomy: groups -> topics, 2026 count, share change since 2024
  map-*.tex    crowding x momentum scatter
  cycle-*.tex  attention cycle (hype-cycle analogy), topics listed per phase

Layout protocol (no overlaps by construction):
  * cards and tables are laid out by TeX itself (nodes holding tabulars, placed relative to each other);
  * scatter labels are placed by a greedy search over candidate positions, using label sizes
    measured by XeLaTeX (not estimated), with dots, labels, reference lines and reserved
    areas as weighted obstacles; a label pushed away from its dot gets a thin leader.
"""
import csv, math, os, pathlib, re, subprocess, tempfile
T = pathlib.Path(__file__).parent.resolve(); D = T / "data"
HEAD = "\\documentclass[border=8pt]{standalone}\n\\input{pt-style}\n\\begin{document}\n"
CM = 0.0351459804  # cm per TeX pt
DOTC = 0.24        # clearance (cm) a label keeps from any other topic's dot

def esc(s): return s.replace("&", "\\&")
def load(name): return list(csv.DictReader(open(D / name)))

SHORT = {
    # CVPR
    "text-rich: OCR, documents, charts": "OCR, documents, charts", "agents, tools & thinking with images": "agents & thinking with images",
    "reasoning efficiency & adaptive thinking": "reasoning efficiency", "geometry, diagrams & visual math": "geometry & visual math",
    "reasoning in generation & editing": "reasoning in generation", "hallucination & faithfulness": "hallucination",
    "spatial & 3D understanding": "spatial & 3D", "video & temporal reasoning": "video & temporal", "embodied planning & VLA": "embodied & VLA",
    "physical & causal reasoning": "physical & causal", "social, affective & intent": "social & affective",
    "prediction & world models": "world models", "abstract puzzles & games": "abstract puzzles", "counting & numerosity": "counting",
    "medical & scientific": "medical & science", "chain-of-thought & multi-step": "chain-of-thought",
    # ACL
    "long reasoning & test-time scaling": "long reasoning & test-time", "reasoning efficiency & overthinking": "reasoning efficiency",
    "compression, quantization & KV cache": "compression & KV cache", "preference alignment & reward models": "preference alignment",
    "knowledge, memory & editing": "knowledge & memory", "dialogue, persona & role-play": "dialogue & persona",
    "synthetic data & data selection": "synthetic data", "culture, values & social": "culture & values",
    "legal, finance & education": "legal, finance, education", "RL post-training (RLVR, GRPO)": "RL post-training",
    "hallucination & factuality": "hallucination", "commonsense & multi-hop QA": "commonsense & multi-hop",
    "retrieval, search & RAG": "retrieval & RAG", "causal, temporal & spatial": "causal, temporal, spatial",
    "multilingual & translation": "multilingual & MT", "uncertainty & calibration": "uncertainty",
    "evaluation, benchmarks & LLM judges": "evaluation & benchmarks", "diversity & homogenization": "diversity",
}
def short(t): return SHORT.get(t, t)

# ------------------------------------------------------------------ measure label boxes with XeLaTeX
def measure(labels, size):
    """Return {label: (width, height, depth)} in cm, set in `size` (e.g. 'small') with the deck font."""
    labels = sorted(set(labels))
    body = "".join(f"\\setbox\\mb\\hbox{{\\{size} {esc(l)}}}\\typeout{{MEAS:{i}:\\the\\wd\\mb:\\the\\ht\\mb:\\the\\dp\\mb}}\n"
                   for i, l in enumerate(labels))
    doc = "\\documentclass{standalone}\n\\input{pt-style}\n\\newbox\\mb\n\\begin{document}\n" + body + "x\n\\end{document}\n"
    with tempfile.TemporaryDirectory() as tmp:
        (pathlib.Path(tmp) / "m.tex").write_text(doc)
        env = dict(os.environ, TEXINPUTS=f"{T}:", max_print_line="10000")
        subprocess.run(["xelatex", "-interaction=nonstopmode", "m.tex"], cwd=tmp, env=env, capture_output=True, check=True)
        log = (pathlib.Path(tmp) / "m.log").read_text()
    out = {}
    for i, w, h, d in re.findall(r"MEAS:(\d+):([\d.]+)pt:([\d.]+)pt:([\d.]+)pt", log):
        out[labels[int(i)]] = (float(w) * CM, float(h) * CM, float(d) * CM)
    assert len(out) == len(labels), "measurement incomplete"
    return out

def gtext(g):
    g = float(g)
    if g >= 1.5: return f"\\textcolor{{ferndark}}{{\\textbf{{×{g:.2f}}}}}"
    if g < 0.8: return f"\\textcolor{{grey500}}{{×{g:.2f}\\,$\\downarrow$}}"
    return f"\\textcolor{{grey700}}{{×{g:.2f}}}"

# ------------------------------------------------------------------ taxonomy (TeX lays out every card)
def taxonomy(rows, fname, ncol):
    groups = []
    for r in rows:
        if not groups or groups[-1][0] != r["group"]: groups.append((r["group"], []))
        groups[-1][1].append(r)
    # keep reading order (top-to-bottom, left-to-right); choose the split that minimises the tallest column
    import itertools
    hts = [len(g[1]) + 1.6 for g in groups]; best = None
    for cuts in itertools.combinations(range(1, len(groups)), ncol - 1):
        b = (0,) + cuts + (len(groups),); m = max(sum(hts[b[i]:b[i + 1]]) for i in range(ncol))
        if best is None or m < best[0]: best = (m, b)
    b = best[1]; cols = [groups[b[i]:b[i + 1]] for i in range(ncol)]
    L = max(w for w, _, _ in measure([short(r["topic"]) for r in rows], "small").values()) + 0.1
    nmax = max(int(r["n2026"]) for r in rows); barw = 1.3
    out = [HEAD, "\\begin{tikzpicture}[font=\\small, card/.style={draw=grey300, line width=0.7pt, rounded corners=4pt, "
                 "fill=white, inner xsep=7pt, inner ysep=6pt, anchor=north west, align=left}]\n"]
    k = 0
    for ci, col in enumerate(cols):
        for gi, (gname, items) in enumerate(col):
            lines = [f"\\multicolumn{{4}}{{@{{}}l@{{}}}}{{\\textbf{{\\color{{ink}}{esc(gname)}}}}}\\\\[2pt]"]
            for r in items:
                n = int(r["n2026"]); w = max(0.03, barw * n / nmax)
                lines.append(f"\\makebox[{L:.2f}cm][l]{{\\color{{grey700}}{esc(short(r['topic']))}}} & "
                             f"\\makebox[0.75cm][r]{{\\color{{grey700}}{n}}} & "
                             f"\\makebox[{barw + 0.05:.2f}cm][l]{{\\textcolor{{grey300}}{{\\rule[0.15ex]{{{w:.3f}cm}}{{1.2ex}}}}}} & "
                             f"\\makebox[1.35cm][r]{{{gtext(r['growth_24_26'])}}}\\\\")
            tab = "\\begin{tabular}{@{}l@{\\hspace{3pt}}l@{\\hspace{6pt}}l@{}l@{}}" + "\n".join(lines) + "\\end{tabular}"
            if gi == 0:
                pos = "(0,0)" if ci == 0 else f"([xshift=0.35cm]top{ci - 1}.north east)"
            else:
                pos = f"([yshift=-0.3cm]g{k - 1}.south west)"
            name = f"g{k}"
            out.append(f"\\node[card] ({name}) at {pos} {{{tab}}};\n")
            if gi == 0: out.append(f"\\coordinate (top{ci}) at ({name}.north east);\n")
            k += 1
    out.append("\\end{tikzpicture}\n\\end{document}\n")
    (T / fname).write_text("".join(out))

# ------------------------------------------------------------------ crowding x momentum scatter
def overlap(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))

def seg_cross(p1, p2, p3, p4):
    def o(a, b, c): return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return o(p1, p2, p3) * o(p1, p2, p4) < 0 and o(p3, p4, p1) * o(p3, p4, p2) < 0

def seg_box(p, q, box):
    """Exact test: does segment p-q pass through the rectangle (Liang-Barsky clipping)?"""
    t0, t1 = 0.0, 1.0; dx, dy = q[0] - p[0], q[1] - p[1]
    for pk, qk in ((-dx, p[0] - box[0]), (dx, box[2] - p[0]), (-dy, p[1] - box[1]), (dy, box[3] - p[1])):
        if pk == 0:
            if qk < 0: return False
        else:
            t = qk / pk
            if pk < 0: t0 = max(t0, t)
            else: t1 = min(t1, t)
            if t0 > t1: return False
    return t1 - t0 > 1e-6

def seg_point(p, q, c):
    """Distance from point c to segment p-q."""
    dx, dy = q[0] - p[0], q[1] - p[1]; L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((c[0] - p[0]) * dx + (c[1] - p[1]) * dy) / L2))
    return math.dist(c, (p[0] + t * dx, p[1] + t * dy))

def place_labels(pts, sizes, obst, bounds, seed=0, steps=120000):
    """Choose one candidate per label minimising overlaps, leader length and crossings (simulated annealing)."""
    import random
    rng = random.Random(seed); pad = 0.07
    cands = []
    for i, ((px, py), (w, h, d)) in enumerate(zip(pts, sizes)):
        ci = []
        for rad in (0.16, 0.45, 0.75, 1.05, 1.4, 1.8):
            for k in range(32):
                th = 2 * math.pi * k / 32; c, s_ = math.cos(th), math.sin(th)
                ax, ay = px + rad * c, py + rad * s_
                x0 = ax if c > 0.3 else (ax - w if c < -0.3 else ax - w / 2)
                y0 = ay if s_ > 0.3 else (ay - h - d if s_ < -0.3 else ay - (h + d) / 2)
                box = (x0 - pad, y0 - pad, x0 + w + pad, y0 + h + d + pad)
                if box[0] < bounds[0] or box[2] > bounds[2] or box[1] < bounds[1] or box[3] > bounds[3]: continue
                lead = None
                if rad > 0.3:
                    qx, qy = min(max(px, x0), x0 + w), min(max(py, y0), y0 + h + d)
                    L = math.dist((px, py), (qx, qy)); ux, uy = (qx - px) / L, (qy - py) / L
                    lead = ((px + 0.1 * ux, py + 0.1 * uy), (qx, qy))
                cost = sum(overlap(box, b) * wt for b, wt in obst if b != (px - DOTC, py - DOTC, px + DOTC, py + DOTC))
                cost += 1.2 * rad + (0.8 if lead else 0.0)
                if lead:
                    cost += sum(300.0 for j, o in enumerate(pts) if j != i and seg_point(lead[0], lead[1], o) < 0.18)
                ext = box if not lead else (min(box[0], lead[0][0]), min(box[1], lead[0][1]), max(box[2], lead[0][0]), max(box[3], lead[0][1]))
                ci.append((cost, box, x0, y0, lead, ext))
        cands.append(ci)
    def pair(a, b):
        ea, eb = a[5], b[5]
        if ea[2] < eb[0] or eb[2] < ea[0] or ea[3] < eb[1] or eb[3] < ea[1]: return 0.0   # far apart: no interaction
        c = overlap(a[1], b[1]) * 20000.0
        if a[4] and b[4] and seg_cross(a[4][0], a[4][1], b[4][0], b[4][1]): c += 300.0
        if a[4] and seg_box(a[4][0], a[4][1], b[1]): c += 300.0
        if b[4] and seg_box(b[4][0], b[4][1], a[1]): c += 300.0
        return c
    n = len(pts); cur = [min(range(len(c)), key=lambda k: c[k][0]) for c in cands]
    def local(i, k): return cands[i][k][0] + sum(pair(cands[i][k], cands[j][cur[j]]) for j in range(n) if j != i)
    total = sum(cands[i][cur[i]][0] for i in range(n)) + sum(pair(cands[i][cur[i]], cands[j][cur[j]]) for i in range(n) for j in range(i + 1, n))
    best, bestc = list(cur), total
    for t in range(steps):
        T = 40.0 * (1 - t / steps) ** 2 + 0.02
        i = rng.randrange(n); k = rng.randrange(len(cands[i]))
        d = local(i, k) - local(i, cur[i])
        if d < 0 or rng.random() < math.exp(-d / T):
            cur[i] = k; total += d
            if total < bestc - 1e-9: best, bestc = list(cur), total
    for i in range(n):  # report anything still colliding
        for j in range(i + 1, n):
            if pair(cands[i][best[i]], cands[j][best[j]]) > 0: print(f"  warn: labels {i} and {j} collide (overlap or leader)")
    print(f"  placement cost {bestc:.1f} (seed {seed})")
    return [cands[i][best[i]][1:5] for i in range(n)], bestc

def scatter(rows, fname, col, W=17.0, H=9.5, size="small"):
    xs = [int(r["n2026"]) for r in rows]; ys = [float(r["growth_24_26"]) for r in rows]
    xmin, xmax = min(xs) * 0.6, max(xs) * 1.7
    ymin, ymax = min(0.45, min(ys) * 0.82), max(ys) * 1.45
    X = lambda v: W * (math.log(v) - math.log(xmin)) / (math.log(xmax) - math.log(xmin))
    Y = lambda v: H * (math.log(v) - math.log(ymin)) / (math.log(ymax) - math.log(ymin))
    med = sorted(xs)[len(xs) // 2]
    labs = [short(r["topic"]) for r in rows]; M = measure(labs + ["emerging", "rising fast"], size)
    pts = [(X(x), Y(y)) for x, y in zip(xs, ys)]
    # obstacles: (box, weight)
    obst = [((px - DOTC, py - DOTC, px + DOTC, py + DOTC), 5000.0) for px, py in pts]
    for yv in (1.0, 1.5): obst.append(((0, Y(yv) - 0.03, W, Y(yv) + 0.03), 600.0))
    obst.append(((X(med) - 0.03, 0, X(med) + 0.03, H), 600.0))
    # reserved corner labels
    ew, eh, _ = M["emerging"]; rw, rh, _ = M["rising fast"]
    obst.append(((0, H - 0.75, ew + 0.4, H), 400.0)); obst.append(((W - rw - 0.4, H - 0.75, W, H), 400.0))
    out = [HEAD, "\\begin{tikzpicture}[font=\\small]\n"]
    out.append(f"\\draw[grey400, line width=0.6pt] (0,{H}) -- (0,0) -- ({W},0);\n")
    for v in (1, 3, 10, 30, 100, 300, 1000):
        if xmin <= v <= xmax:
            out.append(f"\\draw[grey300, line width=0.3pt] ({X(v):.3f},0) -- ({X(v):.3f},{H}); "
                       f"\\node[below, font=\\footnotesize, text=grey700] at ({X(v):.3f},0) {{{v}}};\n")
    for v in (0.5, 1, 2, 4, 8):
        if ymin <= v <= ymax:
            style = "grey500, dashed, line width=0.6pt" if v == 1 else "grey300, line width=0.3pt"
            out.append(f"\\draw[{style}] (0,{Y(v):.3f}) -- ({W},{Y(v):.3f}); "
                       f"\\node[left, font=\\footnotesize, text=grey700] at (0,{Y(v):.3f}) {{×{v:g}}};\n")
    out.append(f"\\draw[{col}, densely dotted, line width=0.9pt] (0,{Y(1.5):.3f}) -- ({W},{Y(1.5):.3f});\n")
    out.append(f"\\draw[grey500, densely dotted, line width=0.9pt] ({X(med):.3f},0) -- ({X(med):.3f},{H});\n")
    out.append(f"\\node[above, font=\\footnotesize, text=grey700] at ({X(med):.3f},{H}) {{median topic size ({med})}};\n")
    out.append(f"\\node[anchor=north west, font=\\small\\bfseries, text=ferndark] at (0.2,{H - 0.15}) {{emerging}};\n")
    out.append(f"\\node[anchor=north east, font=\\small\\bfseries, text=ferndark] at ({W - 0.2},{H - 0.15}) {{rising fast}};\n")
    # right-margin bands: share change since 2024
    bx = W + 0.35
    for lo, hi, name, desc, tc in ((Y(1.5), H, "gaining", "×1.5 or more", "ferndark"),
                                   (Y(0.8), Y(1.5), "flat", "crowded if large,\\\\steady otherwise", "ink"),
                                   (0, Y(0.8), "losing", "below ×0.8", "grey700")):
        out.append(f"\\draw[grey400, line width=0.6pt] ({bx + 0.12},{lo + 0.08:.3f}) -- ({bx},{lo + 0.08:.3f}) -- ({bx},{hi - 0.08:.3f}) -- ({bx + 0.12},{hi - 0.08:.3f});\n")
        out.append(f"\\node[anchor=west, align=left, font=\\footnotesize, text=grey700] at ({bx + 0.25},{(lo + hi) / 2:.3f}) "
                   f"{{\\textbf{{\\color{{{tc}}}{name}}}\\\\{desc}}};\n")
    out.append(f"\\node[below, font=\\small, text=grey700] at ({W / 2},-0.6) {{papers in 2026 (log scale)}};\n")
    out.append(f"\\node[rotate=90, font=\\small, text=grey700] at (-1.3,{H / 2}) {{share in 2026 vs 2024 (log scale)}};\n")
    # label placement: candidates around each dot, then simulated annealing over the joint choice
    sol = min((place_labels(pts, [M[l] for l in labs], obst, (0.05, 0.05, W - 0.02, H - 0.02), seed=k) for k in range(4)),
              key=lambda r: r[1])[0]
    for i, (box, x0, y0, lead) in enumerate(sol):
        if lead:
            (ax_, ay_), (qx, qy) = lead
            out.append(f"\\draw[grey400, line width=0.45pt] ({ax_:.3f},{ay_:.3f}) -- ({qx:.3f},{qy:.3f});\n")
        out.append(f"\\node[anchor=base west, inner sep=0pt, outer sep=0pt, fill=white, font=\\{size}, text=grey700] at ({x0:.3f},{y0 + M[labs[i]][2]:.3f}) {{{esc(labs[i])}}};\n")
    for i, r in enumerate(rows):  # dots on top
        px, py = pts[i]; fill = "white" if xs[i] < 10 else col
        out.append(f"\\fill[{fill}, draw={col}, line width=1pt] ({px:.3f},{py:.3f}) circle (2.6pt);\n")
    hollow = (f" \\quad \\tikz\\draw[{col}, fill=white, line width=1pt] (0,0) circle (2.6pt); fewer than 10 (read with care)"
              if min(xs) < 10 else "")
    out.append(f"\\node[anchor=north west, font=\\footnotesize, text=grey700] at (0,-1.25) "
               f"{{\\tikz\\fill[{col}] (0,0) circle (2.6pt); one topic (10 or more papers in 2026){hollow}}};\n")
    out.append("\\end{tikzpicture}\n\\end{document}\n")
    (T / fname).write_text("".join(out))

# ------------------------------------------------------------------ attention cycle (hype-cycle analogy)
PHASES = [("emerging", "small, share rising", 1.0),
          ("rising fast", "large, share rising", 2.7),
          ("crowded", "largest, share flat", 3.7),
          ("cooling", "share fell in 2026", 2.3),
          ("declining", "share fell since 2024", 0.9),
          ("steady", "mid-size, share flat", 1.8)]

def cycle(rows, fname, col):
    out = [HEAD, "\\begin{tikzpicture}[font=\\small, card/.style={draw=grey300, line width=0.7pt, rounded corners=4pt, "
                 "fill=white, inner xsep=7pt, inner ysep=6pt, anchor=north west, align=left}]\n"]
    for k, (name, desc, hgt) in enumerate(PHASES):
        items = sorted([r for r in rows if r["phase"] == name], key=lambda r: -int(r["n2026"]))
        body = [f"\\textbf{{\\color{{ink}}{name}}}\\\\[-1pt]{{\\footnotesize\\color{{grey500}}{desc}}}\\\\[5pt]"]
        for r in items:
            body.append(f"{{\\color{{grey700}}{esc(short(r['topic']))}}}\\;{{\\footnotesize\\color{{grey500}}{r['n2026']}}}\\\\")
        if not items: body.append("{\\color{grey400}none}\\\\")
        txt = "".join(body).rstrip("\\")
        pos = "(0,0)" if k == 0 else f"([xshift=0.3cm]c{k - 1}.north east)"
        out.append(f"\\node[card] (c{k}) at {pos} {{{txt}}};\n")
        out.append(f"\\coordinate (s{k}) at ([yshift={hgt}cm]c{k}.north);\n")
    out.append("\\coordinate (s-) at ([xshift=-0.1cm, yshift=0.45cm]c0.north west);\n")
    out.append("\\coordinate (s+) at ([yshift=1.9cm]c5.north east);\n")
    out.append(f"\\draw[{col}, line width=2pt] plot[smooth, tension=0.55] coordinates {{(s-) " + " ".join(f"(s{k})" for k in range(6)) + " (s+)};\n")
    for k in range(6):
        out.append(f"\\draw[grey300, line width=0.6pt] ([yshift=-3.2pt]s{k}) -- (c{k}.north);\n")
        out.append(f"\\fill[white, draw={col}, line width=1.2pt] (s{k}) circle (3.2pt);\n")
    out.append("\\node[anchor=south west, font=\\footnotesize, text=grey500] at ([yshift=3.6cm]c0.north west) {attention $\\uparrow$};\n")
    few = sorted([r for r in rows if r["phase"] == "too few to call"], key=lambda r: -int(r["n2026"]))
    if few:
        lst = ", ".join(f"{esc(short(r['topic']))} ({r['n2026']})" for r in few)
        out.append(f"\\node[anchor=north west, font=\\footnotesize, text=grey500] at ([yshift=-0.3cm]current bounding box.south west) "
                   f"{{too few papers to place (under 10, share not rising): {lst}}};\n")
    out.append("\\end{tikzpicture}\n\\end{document}\n")
    (T / fname).write_text("".join(out))

if __name__ == "__main__":
    import sys
    want = set(sys.argv[1:])   # e.g. map-cvpr tax-acl; none = all six
    cv, ac = load("spaces_cvpr.csv"), load("spaces_acl.csv")
    jobs = {"tax-cvpr": lambda: taxonomy(cv, "tax-cvpr.tex", 3), "tax-acl": lambda: taxonomy(ac, "tax-acl.tex", 3),
            "map-cvpr": lambda: scatter(cv, "map-cvpr.tex", "cvpr"), "map-acl": lambda: scatter(ac, "map-acl.tex", "acl"),
            "cycle-cvpr": lambda: cycle(cv, "cycle-cvpr.tex", "cvpr"), "cycle-acl": lambda: cycle(ac, "cycle-acl.tex", "acl")}
    for name, job in jobs.items():
        if not want or name in want: job()
    print("space figures written")
