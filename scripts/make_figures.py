"""Regenerate the two paper figures from results/tables/table_deployment.csv.

Run from the repository root: python scripts/make_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator
from matplotlib.transforms import Bbox
import numpy as np
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6})
INK, INK_2, GRID, FILL = "#0b0b0b", "#52514e", "#e4e3df", "#f4f3f0"

# ------------------------------------------------------------------ Figure 1: pipeline
fig = plt.figure(figsize=(6.7, 2.2))
ax = fig.add_axes([0.005, 0.02, 0.99, 0.96])      # full-width axes so the boxes get the whole figure width
ax.set_xlim(0, 100)
ax.set_ylim(0, 36)
ax.axis("off")
boxes = [
    ("Public datasets", ["TWIFIL (X)", "YouTube comments", "NArabizi (Arabizi)"]),
    ("Audit, splits", ["empty texts", "label conflicts", "duplicates", "cross-split leaks"]),
    ("Training", ["3 TF-IDF models", "4 transformers", "3 seeds, T4 GPU", "dev-set selection"]),
    ("Formats", ["PyTorch fp32", "ONNX fp32", "ONNX int8", "int8 per-ch. + RR"]),
    ("CPU benchmark", ["fresh process", "one thread", "size, peak RSS", "cold start, p50/p95", "full-test F1"]),
    ("Analysis", ["Pareto frontiers", "budget tiers", "int8 audit", "shortcut test"]),
]
n = len(boxes)
w, gap, y0, h = 14.6, 2.2, 1.5, 32
x = 0.6
lefts = []
for title, lines in boxes:
    ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.2,rounding_size=1.0",
                                linewidth=0.7, edgecolor=INK_2, facecolor=FILL))
    ax.text(x + w / 2, y0 + h - 4.0, title, ha="center", va="center", fontsize=6.8, fontweight="bold", color=INK)
    for k, line in enumerate(lines):
        ax.text(x + w / 2, y0 + h - 10.2 - k * 4.6, line, ha="center", va="center", fontsize=5.9, color=INK_2)
    lefts.append(x)
    x += w + gap
for i in range(n - 1):
    xa, xb = lefts[i] + w + 0.25, lefts[i + 1] - 0.25
    ax.annotate("", xy=(xb, y0 + h / 2), xytext=(xa, y0 + h / 2),
                arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=0.8, shrinkA=0, shrinkB=0, mutation_scale=6))
fig.savefig(os.path.join(FIG_DIR, "fig1_pipeline.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ------------------------------------------------------------------ Figure 2: Pareto frontiers
D = pd.read_csv(os.path.join(ROOT, "results", "tables", "table_deployment.csv"))
D = D[~D["format"].str.contains("no RR")].copy()
FAMILY = {"NB-char": "Classical (TF-IDF)", "SVM-char": "Classical (TF-IDF)", "LR-word+char": "Classical (TF-IDF)",
          "mE5-small": "Multilingual small transformer", "DziriBERT": "Arabic/dialect BERT",
          "MARBERTv2": "Arabic/dialect BERT", "CAMeLBERT-DA": "Arabic/dialect BERT"}
COLOUR = {"Classical (TF-IDF)": "#2a78d6", "Multilingual small transformer": "#eb6834", "Arabic/dialect BERT": "#1baf7a"}
MARKER = {"scikit-learn": "o", "PyTorch fp32": "o", "ONNX fp32": "s", "ONNX int8": "^", "ONNX int8 per-channel+RR": "D"}
TITLES = {("twifil_clean", 0): "(a) TWIFIL, cleaned split: latency", ("twifil_clean", 1): "(b) TWIFIL, cleaned split: size",
          ("youtube", 0): "(c) Algerian Dialect YouTube: latency", ("youtube", 1): "(d) Algerian Dialect YouTube: size"}
LAT, SIZE = "p50 (ms)", "size (MB)"
RR = "ONNX int8 per-channel+RR"
# Hand-placed labels for the Pareto-frontier points: (dataset, axis, model, format) -> (dx, dy, ha) in points.
# Frontier points of the same model that overlap (int8 default and int8-RR) share one label, anchored on the more
# accurate point; the marker shape tells the format.
LABELS = {
    ("twifil_clean", LAT, "SVM-char", "scikit-learn"): (4, 6, "left"),
    ("twifil_clean", LAT, "LR-word+char", "scikit-learn"): (4, 6, "left"),
    ("twifil_clean", LAT, "mE5-small", RR): (-7, -11, "right"),
    ("twifil_clean", LAT, "DziriBERT", RR): (-4, 7, "right"),
    ("twifil_clean", LAT, "DziriBERT", "ONNX fp32"): (0, 8, "center"),
    ("twifil_clean", SIZE, "SVM-char", "scikit-learn"): (-5, 3, "right"),
    ("twifil_clean", SIZE, "LR-word+char", "scikit-learn"): (4, 6, "left"),
    ("twifil_clean", SIZE, "DziriBERT", RR): (-4, 7, "right"),
    ("twifil_clean", SIZE, "DziriBERT", "PyTorch fp32"): (-2, 8, "right"),
    ("youtube", LAT, "SVM-char", "scikit-learn"): (4, 6, "left"),
    ("youtube", LAT, "MARBERTv2", "ONNX int8"): (-7, -4, "right"),
    ("youtube", LAT, "DziriBERT", RR): (-4, 6, "right"),
    ("youtube", LAT, "DziriBERT", "ONNX fp32"): (0, 8, "center"),
    ("youtube", SIZE, "SVM-char", "scikit-learn"): (4, 6, "left"),
    ("youtube", SIZE, "DziriBERT", RR): (-4, 6, "right"),
    ("youtube", SIZE, "DziriBERT", "PyTorch fp32"): (-2, 8, "right"),
}
XLIM_LEFT = {("twifil_clean", SIZE): 1.0}


def pareto(cost, qual):
    cost, qual = np.asarray(cost), np.asarray(qual)
    keep = np.ones(len(cost), bool)
    for i in range(len(cost)):
        dom = (cost <= cost[i]) & (qual >= qual[i]) & ((cost < cost[i]) | (qual > qual[i]))
        keep[i] = not dom.any()
    return keep


fig_h = 6.4
fig, axes = plt.subplots(2, 2, figsize=(6.7, fig_h))
for i, ds in enumerate(["twifil_clean", "youtube"]):
    b = D[D.dataset == ds].reset_index(drop=True)
    for j, (col, lab) in enumerate([(LAT, "Latency p50, batch 1, one thread (ms, log)"),
                                    (SIZE, "Size on disk (MiB, log)")]):
        ax = axes[i][j]
        m = pareto(b[col], b["macro-F1 (%)"])
        f = b[m].sort_values(col)
        ax.step(f[col], f["macro-F1 (%)"], where="post", color=INK_2, lw=1.0, zorder=1)
        for mdl, fmt_, xv, yv in zip(b["model"], b["format"], b[col], b["macro-F1 (%)"]):
            ax.scatter(xv, yv, s=34, marker=MARKER[fmt_], color=COLOUR[FAMILY[mdl]],
                       edgecolor="white", linewidth=1.0, zorder=3)
        # labels: frontier points only, one per overlapping same-model pair
        labelled = 0
        for mdl, fmt_, xv, yv in zip(f["model"], f["format"], f[col], f["macro-F1 (%)"]):
            key = (ds, col, mdl, fmt_)
            if key not in LABELS:
                continue
            dx, dy, ha = LABELS[key]
            ax.annotate(mdl, (xv, yv), xytext=(dx, dy), textcoords="offset points", fontsize=6, color=INK_2,
                        ha=ha, va="bottom" if dy > 0 else "top",
                        arrowprops=dict(arrowstyle="-", color="#a8a7a2", lw=0.4, shrinkA=0, shrinkB=3))
            labelled += 1
        frontier_models = f.groupby("model").size()
        ax.set_xscale("log")
        if (ds, col) in XLIM_LEFT:
            ax.set_xlim(left=XLIM_LEFT[(ds, col)])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel(lab, fontsize=7, color=INK_2)
        ax.set_ylabel("Macro-F1 (%)", fontsize=7, color=INK_2)
        ax.set_title(TITLES[(ds, j)], fontsize=7.4, color=INK, loc="left")
        ax.grid(True, which="major", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(which="both", labelsize=6.3, colors=INK_2, length=2)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color(GRID)
        ys = b["macro-F1 (%)"]
        ax.set_ylim(ys.min() - 1.5, ys.max() + 1.8)
        print(ds, col, "frontier points:", len(f), "labels:", labelled)
fam = [Line2D([], [], ls="", marker="o", ms=5, color=c, markeredgecolor="white", label=k) for k, c in COLOUR.items()]
fmt = [Line2D([], [], ls="", marker=mk, ms=5, color=INK_2, markeredgecolor="white", label=k)
       for k, mk in (("native (scikit-learn / PyTorch fp32)", "o"), ("ONNX fp32", "s"), ("ONNX int8, default", "^"),
                     ("ONNX int8, per-channel + reduce-range", "D"))]
fr = [Line2D([], [], color=INK_2, lw=1.0, label="Pareto frontier")]
fig.tight_layout(rect=(0, 0.75 / fig_h, 1, 1), h_pad=1.6, w_pad=1.6)
fig.legend(handles=fam + fmt + fr, loc="lower center", ncol=3, fontsize=6.3, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.savefig(os.path.join(FIG_DIR, "fig2_pareto.png"), dpi=300, bbox_inches="tight", facecolor="white")
print("figures written")
