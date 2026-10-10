"""Write the chart .tex files from the CSVs in tikz/data (numbers are computed upstream, never typed)."""
import csv, pathlib
D = pathlib.Path(__file__).parent / "data"; T = pathlib.Path(__file__).parent
def rows(name): return list(csv.DictReader(open(D / name)))
HEAD = "\\documentclass[border=8pt]{standalone}\n\\input{pt-style}\n\\begin{document}\n"
# ---- C1: title-topic trends, small multiples (CVPR vs ACL, 2024-2026)
tr = rows("d1_title_trends.csv")
topics = ["RL post-training", "reasoning", "agents", "names a language model", "benchmark / dataset", "distillation"]
out = [HEAD, "\\begin{tikzpicture}[font=\\small]\n",
       "\\begin{groupplot}[group style={group size=3 by 2, horizontal sep=1.9cm, vertical sep=1.75cm}, surveyaxis,",
       " width=6.0cm, height=4.3cm, xmin=2024, xmax=2026, tick label style={font=\\small, text=grey700}, xtick={2024,2025,2026}, xticklabels={2024,2025,2026},",
       " ymin=0, yticklabel={\\pgfmathprintnumber{\\tick}\\%}, title style={font=\\normalsize\\bfseries, text=ink, yshift=-2pt}]\n"]
for t in topics:
    pts = {v: [(int(r["year"]), float(r["share_pct"])) for r in tr if r["venue"] == v and r["topic"] == t] for v in ("CVPR", "ACL")}
    ymax = max(y for v in pts for _, y in pts[v]); ymax = max(1.0, ymax * 1.18)
    out.append(f"\\nextgroupplot[title={{{t}}}, ymax={ymax:.2f}]\n")
    for v, col in (("CVPR", "cvpr"), ("ACL", "acl")):
        coords = " ".join(f"({x},{y})" for x, y in pts[v])
        out.append(f"\\addplot[{col}, line width=1.6pt, mark=*, mark size=2.2pt, mark options={{fill={col}, draw=white, line width=0.6pt}}] coordinates {{{coords}}};\n")
        x, y = pts[v][-1]
        out.append(f"\\node[anchor=west, font=\\small, text=grey700] at (axis cs:{x},{y}) {{\\,{y:.1f}\\%}};\n")
out.append("\\end{groupplot}\n")
out.append("\\node[anchor=south west, font=\\small] at ($(group c1r1.north west)+(-0.2,0.75)$) {\\tikz\\draw[cvpr, line width=1.6pt] (0,0)--(0.45,0); CVPR\\quad \\tikz\\draw[acl, line width=1.6pt] (0,0)--(0.45,0); ACL\\quad\\textcolor{grey500}{share of all accepted main-conference papers whose \\emph{title} mentions the topic}};\n")
out.append("\\end{tikzpicture}\n\\end{document}\n")
(T / "title-trends.tex").write_text("".join(out))
# ---- C2: RL share among in-scope papers, slope chart 2025 -> 2026
m = {r["venue_year"]: r for r in rows("d2_methods.csv")}
def share(k): return 100 * int(m[k]["RL"]) / int(m[k]["n_scope"])
c25, c26, a25, a26 = share("CVPR 2025"), share("CVPR 2026"), share("ACL 2025"), share("ACL 2026")
out = [HEAD, "\\begin{tikzpicture}[font=\\small]\n",
       "\\begin{axis}[surveyaxis, width=11cm, height=6.4cm, xmin=2025, xmax=2026, xtick={2025,2026}, xticklabels={2025,2026},",
       " ymin=0, ymax=32, ytick={0,10,20,30}, yticklabel={\\pgfmathprintnumber{\\tick}\\%}, enlarge x limits={abs=0.8},",
       " ylabel={share of papers using RL post-training}]\n"]
for col, a, b, lab, n0, n1 in (("cvpr", c25, c26, "CVPR (language-model papers)", m["CVPR 2025"], m["CVPR 2026"]),
                                ("acl", a25, a26, "ACL (all main papers)", m["ACL 2025"], m["ACL 2026"])):
    out.append(f"\\addplot[{col}, line width=2pt, mark=*, mark size=2.8pt, mark options={{fill={col}, draw=white, line width=0.7pt}}] coordinates {{(2025,{a:.2f}) (2026,{b:.2f})}};\n")
    anc = "south east" if col == "cvpr" else "north east"
    out.append(f"\\node[anchor={anc}, font=\\footnotesize, text=grey700] at (axis cs:2025,{a:.2f}) {{{a:.0f}\\% ({n0['RL']} of {n0['n_scope']})\\,}};\n")
    out.append(f"\\node[anchor=west, font=\\footnotesize, text=grey700, align=left] at (axis cs:2026,{b:.2f}) {{\\,\\textbf{{{b:.0f}\\%}} ({n1['RL']} of {n1['n_scope']})\\\\\\,{lab}}};\n")
out.append("\\end{axis}\n\\end{tikzpicture}\n\\end{document}\n")
(T / "rl-share.tex").write_text("".join(out))
# ---- C3: what RL papers target, by primary task, 2026 (two panels)
tk = rows("d4_rl_primary_task.csv")
def panel(name, col, xshift, title):
    ps = sorted([r for r in tk if r["panel"] == name], key=lambda r: float(r["pct"]))
    labels = ",".join("{" + r["task"].replace(",", "{,}") + "}" for r in ps)
    coords = " ".join(f"({float(r['pct'])},{i})" for i, r in enumerate(ps))
    n = ps[0]["n"]
    s = [f"\\begin{{axis}}[at={{({xshift},0)}}, xbar, bar width=7pt, width=6.6cm, height=7.6cm, axis lines=left, axis line style={{grey400}},",
         " xmin=0, xmax=30, xtick=\\empty, xticklabels={}, axis x line=none, axis line style={grey400, -}, ytick={0,...,%d}, yticklabels={%s}," % (len(ps) - 1, labels),
         " y tick label style={font=\\footnotesize, text=grey700, align=right}, ytick style={draw=none}, enlarge y limits=0.06, clip=false,",
         f" title={{{title}}}, title style={{font=\\small\\bfseries, text=ink, align=center}}]\n",
         f"\\addplot[fill={col}, draw=none, rounded corners=1pt] coordinates {{{coords}}};\n"]
    for i, r in enumerate(ps):
        s.append(f"\\node[anchor=west, font=\\footnotesize, text=grey700] at (axis cs:{float(r['pct'])},{i}) {{\\,{float(r['pct']):.0f}\\%}};\n")
    s.append("\\end{axis}\n")
    return "".join(s)
out = [HEAD, "\\begin{tikzpicture}[font=\\small]\n",
       panel("CVPR 2026 (vision RL papers)", "cvpr", "0cm", "CVPR 2026: RL papers on vision (n = 210)"),
       panel("ACL 2026 (text-only RL papers)", "acl", "11.6cm", "ACL 2026: text-only RL papers (n = 336)"),
       "\\end{tikzpicture}\n\\end{document}\n"]
(T / "rl-tasks.tex").write_text("".join(out))
print("charts written")
