"""Generate five candidate versions of Figure 1 from the per-state result file.

Every coordinate is read from s3_expected_ecological.json at build time, so the
variants cannot disagree with the analysis or with each other.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "analyses" / "results" / "s3_expected_ecological.json"

d = json.load(RESULTS.open())
per = d["per_state"]
keys = sorted(per, key=int)
x = [per[k]["prop_elderly"] * 100 for k in keys]
obs = [per[k]["observed_mortality"] * 100 for k in keys]
imp = [per[k]["model_implied_mortality"] * 100 for k in keys]
b_obs, b_imp = d["observed_ecological_slope"], d["model_implied_ecological_slope"]

xlo, xhi = min(x), max(x)
mx, mo, mi = sum(x) / len(x), sum(obs) / len(obs), sum(imp) / len(imp)
line = lambda m, b, a: (f"({a:.4g},{m + b * (a - mx):.4f})")
obs_fit = f"{line(mo, b_obs, xlo)} {line(mo, b_obs, xhi)}"
imp_fit = f"{line(mi, b_imp, xlo)} {line(mi, b_imp, xhi)}"

P = lambda ys: " ".join(f"({a:.4f},{b:.4f})" for a, b in zip(x, ys))

PRE = r"""\documentclass[tikz,border=6pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage[scaled]{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage[T1]{fontenc}
\definecolor{bordblue}{HTML}{3B82F6}
\definecolor{bordorange}{HTML}{D97706}
\definecolor{navy}{HTML}{1E3A5F}
\definecolor{midgray}{HTML}{8A8A8A}
\definecolor{anncolor}{HTML}{6B7280}
\begin{document}
"""
POST = "\n\\end{document}\n"

AX = (f"width=11.5cm, height=7.4cm,\n"
      f"  xlabel={{Share of records aged 70 or older (\\%)}},\n"
      f"  ylabel={{State case fatality (\\%)}},\n"
      f"  xmin={xlo:.4g}, xmax={xhi:.4g}, ymax=15.2,\n"
      f"  axis lines=left, axis line style={{-}}, tick align=outside,\n"
      f"  ymajorgrids=true, grid style={{gray!30, line width=0.3pt}},\n"
      f"  xlabel style={{font=\\small}}, ylabel style={{font=\\small}},\n"
      f"  tick label style={{font=\\footnotesize}},\n"
      f"  legend style={{at={{(0.02,0.98)}}, anchor=north west, draw=none,\n"
      f"                fill=none, font=\\footnotesize, row sep=2pt}},\n"
      f"  legend cell align=left,")

OUT = {}

# A — current palette, restyled axes
OUT["A_restyled"] = f"""\\begin{{tikzpicture}}
\\begin{{axis}}[{AX}]
\\addplot[bordblue, line width=1.1pt, forget plot] coordinates {{{obs_fit}}};
\\addplot[bordorange, line width=1.1pt, dashed, forget plot] coordinates {{{imp_fit}}};
\\addplot[only marks, mark=*, mark size=1.8pt, color=bordblue] coordinates {{{P(obs)}}};
\\addlegendentry{{observed state rate, slope {b_obs:.3f}}}
\\addplot[only marks, mark=o, mark size=1.8pt, color=bordorange] coordinates {{{P(imp)}}};
\\addlegendentry{{rate implied by the record-level model, slope {b_imp:.3f}}}
\\end{{axis}}
\\end{{tikzpicture}}"""

# B — a connector per state, so each pair is unambiguous
seg = "\n".join(
    f"\\draw[anncolor, line width=0.4pt] (axis cs:{a:.4f},{o:.4f}) -- (axis cs:{a:.4f},{m:.4f});"
    for a, o, m in zip(x, obs, imp))
OUT["B_paired"] = f"""\\begin{{tikzpicture}}
\\begin{{axis}}[{AX}]
{seg}
\\addplot[bordblue, line width=1.1pt, forget plot] coordinates {{{obs_fit}}};
\\addplot[bordorange, line width=1.1pt, dashed, forget plot] coordinates {{{imp_fit}}};
\\addplot[only marks, mark=*, mark size=1.8pt, color=bordblue] coordinates {{{P(obs)}}};
\\addlegendentry{{observed state rate, slope {b_obs:.3f}}}
\\addplot[only marks, mark=o, mark size=1.8pt, color=bordorange] coordinates {{{P(imp)}}};
\\addlegendentry{{rate implied by the record-level model, slope {b_imp:.3f}}}
\\end{{axis}}
\\end{{tikzpicture}}"""

# C — two panels, no overlap at all
panel = (AX.replace("width=11.5cm", "width=6.6cm").replace("height=7.4cm", "height=6.4cm"))
OUT["C_panels"] = f"""\\begin{{tikzpicture}}
\\begin{{axis}}[{panel}
  title={{Observed, slope {b_obs:.3f}}}, title style={{font=\\small}}, legend style={{draw=none,fill=none}}]
\\addplot[bordblue, line width=1.1pt, forget plot] coordinates {{{obs_fit}}};
\\addplot[only marks, mark=*, mark size=1.7pt, color=bordblue, forget plot] coordinates {{{P(obs)}}};
\\end{{axis}}
\\begin{{axis}}[{panel}
  at={{(7.4cm,0)}}, ylabel={{}},
  title={{Model-implied, slope {b_imp:.3f}}}, title style={{font=\\small}}]
\\addplot[bordorange, line width=1.1pt, dashed, forget plot] coordinates {{{imp_fit}}};
\\addplot[only marks, mark=o, mark size=1.7pt, color=bordorange, forget plot] coordinates {{{P(imp)}}};
\\end{{axis}}
\\end{{tikzpicture}}"""

# D — the discrepancy plotted directly: one series, no overlap
res = [o - m for o, m in zip(obs, imp)]
mr = sum(res) / len(res)
diff = b_obs - b_imp
res_fit = f"({xlo:.4g},{mr + diff * (xlo - mx):.4f}) ({xhi:.4g},{mr + diff * (xhi - mx):.4f})"
AXD = AX.replace("State case fatality (\\%)",
                 "Observed minus model-implied (points)").replace("ymax=15.2, ", "")
OUT["D_residual"] = f"""\\begin{{tikzpicture}}
\\begin{{axis}}[{AXD}]
\\draw[anncolor, line width=0.4pt] (axis cs:{xlo:.4g},0) -- (axis cs:{xhi:.4g},0);
\\addplot[bordblue, line width=1.1pt, forget plot] coordinates {{{res_fit}}};
\\addplot[only marks, mark=*, mark size=1.8pt, color=bordblue] coordinates {{{P(res)}}};
\\addlegendentry{{state residual, slope {diff:.3f}}}
\\end{{axis}}
\\end{{tikzpicture}}"""

# E — grayscale-survivable: shape carries the distinction, not hue alone
OUT["E_grayscale"] = f"""\\begin{{tikzpicture}}
\\begin{{axis}}[{AX}]
\\addplot[navy, line width=1.1pt, forget plot] coordinates {{{obs_fit}}};
\\addplot[midgray, line width=1.1pt, dashed, forget plot] coordinates {{{imp_fit}}};
\\addplot[only marks, mark=*, mark size=1.9pt, color=navy] coordinates {{{P(obs)}}};
\\addlegendentry{{observed state rate, slope {b_obs:.3f}}}
\\addplot[only marks, mark=square, mark size=1.9pt, color=midgray] coordinates {{{P(imp)}}};
\\addlegendentry{{rate implied by the record-level model, slope {b_imp:.3f}}}
\\end{{axis}}
\\end{{tikzpicture}}"""

for name, body in OUT.items():
    Path(f"fig1_{name}.tex").write_text(PRE + body + POST)
    print(f"wrote fig1_{name}.tex")

# ---------------------------------------------------------------- round two
ymin, ymax = 3.4, 14.2          # one range, shared by every panel below
AXS = AX.replace("ymax=15.2,", f"ymin={ymin}, ymax={ymax},")

# G — two panels on ONE shared y-range
panelS = AXS.replace("width=11.5cm", "width=6.6cm").replace("height=7.4cm", "height=6.4cm")
OUT["G_panels_shared"] = f"""\\begin{{tikzpicture}}
\\begin{{axis}}[{panelS}
  title={{Observed, slope {b_obs:.3f}}}, title style={{font=\\small}}]
\\addplot[bordblue, line width=1.1pt, forget plot] coordinates {{{obs_fit}}};
\\addplot[only marks, mark=*, mark size=1.7pt, color=bordblue, forget plot] coordinates {{{P(obs)}}};
\\end{{axis}}
\\begin{{axis}}[{panelS}
  at={{(7.2cm,0)}}, ylabel={{}}, yticklabels={{}},
  title={{Model-implied, slope {b_imp:.3f}}}, title style={{font=\\small}}]
\\addplot[bordorange, line width=1.1pt, dashed, forget plot] coordinates {{{imp_fit}}};
\\addplot[only marks, mark=o, mark size=1.7pt, color=bordorange, forget plot] coordinates {{{P(imp)}}};
\\end{{axis}}
\\end{{tikzpicture}}"""

# H — the real data once, with both slopes drawn over it
OUT["H_one_cloud"] = f"""\\begin{{tikzpicture}}
\\begin{{axis}}[{AXS}]
\\addplot[only marks, mark=*, mark size=1.9pt, color=navy] coordinates {{{P(obs)}}};
\\addlegendentry{{observed state rate}}
\\addplot[navy, line width=1.2pt] coordinates {{{obs_fit}}};
\\addlegendentry{{slope across states, {b_obs:.3f}}}
\\addplot[bordorange, line width=1.2pt, dashed] coordinates {{{imp_fit}}};
\\addlegendentry{{slope the record-level model implies, {b_imp:.3f}}}
\\end{{axis}}
\\end{{tikzpicture}}"""

# J — H, with the divergence between the two slopes shaded
OUT["J_wedge"] = f"""\\begin{{tikzpicture}}
\\begin{{axis}}[{AXS}]
\\addplot[name path=A, draw=none, forget plot] coordinates {{{obs_fit}}};
\\addplot[name path=B, draw=none, forget plot] coordinates {{{imp_fit}}};
\\addplot[bordorange!12] fill between[of=A and B];
\\addlegendentry{{the discrepancy, {b_obs - b_imp:.3f} per point of elderly share}}
\\addplot[only marks, mark=*, mark size=1.9pt, color=navy] coordinates {{{P(obs)}}};
\\addlegendentry{{observed state rate}}
\\addplot[navy, line width=1.2pt] coordinates {{{obs_fit}}};
\\addlegendentry{{slope across states, {b_obs:.3f}}}
\\addplot[bordorange, line width=1.2pt, dashed] coordinates {{{imp_fit}}};
\\addlegendentry{{slope the model implies, {b_imp:.3f}}}
\\end{{axis}}
\\end{{tikzpicture}}"""

# K — an arrow per state, from what the model implies to what was observed
arr = "\n".join(
    f"\\draw[-{{Stealth[length=3.2pt,width=2.6pt]}}, anncolor, line width=0.45pt]"
    f" (axis cs:{a:.4f},{m:.4f}) -- (axis cs:{a:.4f},{o:.4f});"
    for a, o, m in zip(x, obs, imp))
OUT["K_arrows"] = f"""\\begin{{tikzpicture}}
\\begin{{axis}}[{AXS}]
{arr}
\\addplot[navy, line width=1.1pt, forget plot] coordinates {{{obs_fit}}};
\\addplot[bordorange, line width=1.1pt, dashed, forget plot] coordinates {{{imp_fit}}};
\\addplot[only marks, mark=*, mark size=1.8pt, color=navy] coordinates {{{P(obs)}}};
\\addlegendentry{{observed state rate, slope {b_obs:.3f}}}
\\addplot[only marks, mark=o, mark size=1.8pt, color=bordorange] coordinates {{{P(imp)}}};
\\addlegendentry{{rate the model implies, slope {b_imp:.3f}}}
\\end{{axis}}
\\end{{tikzpicture}}"""

PRE2 = PRE.replace("\\pgfplotsset{compat=1.18}",
                   "\\pgfplotsset{compat=1.18}\n\\usepgfplotslibrary{fillbetween}\n"
                   "\\usetikzlibrary{arrows.meta}")
for name in ("G_panels_shared", "H_one_cloud", "J_wedge", "K_arrows"):
    Path(f"fig1_{name}.tex").write_text(PRE2 + OUT[name] + POST)
    print(f"wrote fig1_{name}.tex")

# L — two panels, shared y-range, tick labels on BOTH, scaled up
panelL = (AXS.replace("width=11.5cm", "width=7.8cm")
             .replace("height=7.4cm", "height=7.6cm")
          + "\n  ytick={4,6,8,10,12,14},")
OUT["L_panels_labeled"] = f"""\\begin{{tikzpicture}}
\\begin{{axis}}[{panelL}
  title={{Observed, slope {b_obs:.3f}}}, title style={{font=\\small}}]
\\addplot[bordblue, line width=1.2pt, forget plot] coordinates {{{obs_fit}}};
\\addplot[only marks, mark=*, mark size=1.9pt, color=bordblue, forget plot] coordinates {{{P(obs)}}};
\\end{{axis}}
\\begin{{axis}}[{panelL}
  at={{(9.0cm,0)}}, ylabel={{}},
  title={{Model-implied, slope {b_imp:.3f}}}, title style={{font=\\small}}]
\\addplot[bordorange, line width=1.2pt, dashed, forget plot] coordinates {{{imp_fit}}};
\\addplot[only marks, mark=o, mark size=1.9pt, color=bordorange, forget plot] coordinates {{{P(imp)}}};
\\end{{axis}}
\\end{{tikzpicture}}"""
Path("fig1_L_panels_labeled.tex").write_text(PRE2 + OUT["L_panels_labeled"] + POST)
print("wrote fig1_L_panels_labeled.tex")
