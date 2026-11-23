"""Result figures of the paper and its supplementary material, drawn from the results package (EpiRIME-results).

    python scripts/plot_results.py --results path/to/EpiRIME-results --out figures

Main text (Figures 3 to 9):
    fig_convergence      Figure 3  convergence of the twelve algorithms, first function of each function class
    fig_boxplot          Figure 4  final best fitness of the same functions
    fig_ablation         Figure 5  Friedman mean rank of the four versions in the ablation analysis
    fig_sensitivity      Figure 6  Friedman mean rank of EpiRIME for ten values of the nucleus-pool ratio rho
    fig_runtime          Figure 7  mean CPU time per run
    fig_fs_tradeoff      Figure 8  mean number of selected genes and mean test balanced error
    fig_fs_testbox       Figure 9  test balanced error of the best solution of each run
Supplementary material (Figures S1 to S24), one figure per benchmark setting:
    figS_convergence_*   Figures S1-S4      figS_boxplot_*       Figures S5-S8
    figS_abl_convergence_*  Figures S9-S12  figS_abl_boxplot_*   Figures S13-S16
    figS_sens_convergence_* Figures S17-S20 figS_sens_boxplot_*  Figures S21-S24

The dashed reference line of fig_fs_tradeoff (KNN classifier with all 2000 genes) is computed with
epirime/gene_selection.py from the processed datasets in data/geo/.

Requires matplotlib (besides NumPy). Text is set in Times New Roman when the font is installed.
"""
import argparse
import csv
import gzip
import json
import os
import sys
from collections import defaultdict

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, ScalarFormatter  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from epirime.gene_selection import DATASETS, GeneSelection  # noqa: E402

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "ytick.minor.size": 1.5,
    "xtick.direction": "in", "ytick.direction": "in", "pdf.fonttype": 42,
})
META = {"Creator": None, "Producer": None, "CreationDate": None}

CELLS = [("CEC2017", 30), ("CEC2017", 50), ("CEC2022", 10), ("CEC2022", 20)]
FUNCS = {"CEC2017": [1] + list(range(3, 31)), "CEC2022": list(range(1, 13))}
REP = {"CEC2017": [1, 4, 11, 21], "CEC2022": [1, 2, 6, 9]}      # first function of each function class
CATS = ["Unimodal", "Multimodal", "Hybrid", "Composition"]
CELL_COL = ["#2E4E6F", "#5B86B0", "#A3553A", "#D99A6C"]
RHOS = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]

# (name in the results files, label, colour, line style, marker)
BENCH = [("L-SHADE", "L-SHADE", "#1F77B4", "--", "s"), ("BWDE", "BWDE", "#17BECF", "--", "^"),
         ("TCDE", "TCDE", "#2CA02C", "--", "v"), ("DPSDE", "DPSDE", "#9467BD", "--", "D"),
         ("RIME", "RIME", "#7F7F7F", "-", "o"), ("IRIME", "IRIME", "#8C564B", "-", "p"),
         ("HRIME-MSP", "HRIME-MSP", "#E377C2", "-", "h"), ("TERIME", "TERIME", "#BCBD22", "-", "<"),
         ("DNMRIME", "DNMRIME", "#FF7F0E", "-", ">"), ("MRIME-CD", "MRIME-CD", "#A6761D", "-", "P"),
         ("HDRIME", "HDRIME", "#1B9E77", "-", "X"), ("EpiRIME", "EpiRIME", "#C0161E", "-", "o")]
ABL = [("RIME", "RIME", "#7F7F7F", "-", "s"), ("RIME + S1", "RIME + S1", "#1F77B4", "-", "^"),
       ("RIME + S2", "RIME + S2", "#FF7F0E", "-", "v"), ("EpiRIME", "EpiRIME", "#C0161E", "-", "o")]
_cmap = plt.get_cmap("viridis")
SENS = [(f"rho={r:g}", rf"$\rho = {r:g}$",
         "#C0161E" if r == 0.3 else _cmap(k / (len(RHOS) - 1) * 0.92), "-",
         "o" if r == 0.3 else "s^vD<>ph*"[k if k < 6 else k - 1]) for k, r in enumerate(RHOS)]

FS_DS = [("NPC-GSE12452", "NPC-1"), ("NPC-GSE53819", "NPC-2"), ("LSCC-GSE59102", "LSCC-1"),
         ("LSCC-GSE143224", "LSCC-2"), ("HNSCC-GSE107591", "HNSCC"), ("OSCC-GSE30784", "OSCC-1"),
         ("OSCC-GSE25099", "OSCC-2"), ("OSCC-GSE37991", "OSCC-3"), ("OSCC-GSE13601", "OSCC-4")]
FS_ALGS = ["HRO-GWO", "RBNRO-DE", "GOG-MBSHO", "iBABC-CGO", "BHOA-CM", "BPRGO", "AIEEFO", "SPHMO", "EpiRIME"]
FS_COL = {"HRO-GWO": "#0072B2", "RBNRO-DE": "#E69F00", "GOG-MBSHO": "#009E73", "iBABC-CGO": "#CC79A7",
          "BHOA-CM": "#56B4E9", "BPRGO": "#8C564B", "AIEEFO": "#7F7F7F", "SPHMO": "#BCBD22", "EpiRIME": "#D62728"}
FS_MARK = ["o", "s", "^", "v", "D", "P", "X", "<", ">"]


# ---------------------------------------------------------------------------------------------------- data
class Experiment:
    """final_fitness.csv and convergence_<suite>_D<dim>.json.gz of one experiment folder."""

    def __init__(self, folder):
        self.folder = folder
        self.final = defaultdict(dict)
        with open(os.path.join(folder, "final_fitness.csv"), newline="") as fh:
            for r in csv.DictReader(fh):
                key = (r["suite"], int(r["dim"]), r["function"], r["algorithm"])
                self.final[key][int(r["run"])] = float(r["best_fitness"])
        self._conv = {}

    def fitness(self, suite, dim, func, alg):
        runs = self.final[(suite, dim, f"F{func}", alg)]
        assert len(runs) == 30, (suite, dim, func, alg)
        return np.array([runs[k] for k in sorted(runs)])

    def curves(self, suite, dim, func, alg):
        if (suite, dim) not in self._conv:
            with gzip.open(os.path.join(self.folder, f"convergence_{suite}_D{dim}.json.gz"), "rt") as fh:
                self._conv[(suite, dim)] = json.load(fh)
        d = self._conv[(suite, dim)]
        assert d["budget"] == 1000 * dim
        return np.array(d["best_fitness"][alg][f"F{func}"], float)        # (30 runs, 50 points)


def mean_ranks(table):
    """Friedman mean rank of each column of `table` (rows: functions), 1 = smallest; ties get the average rank."""
    ranks = []
    for row in np.asarray(table, float):
        order = np.argsort(row, kind="stable")
        r = np.empty(len(row))
        i = 0
        while i < len(row):
            j = i
            while j + 1 < len(row) and row[order[j + 1]] == row[order[i]]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2 + 1
            i = j + 1
        ranks.append(r)
    return np.mean(ranks, axis=0)


# ---------------------------------------------------------------------------------------------------- helpers
def save(fig, path):
    fig.savefig(f"{path}.pdf", bbox_inches="tight", pad_inches=0.03, metadata=META)
    fig.savefig(f"{path}.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _sci(v):
    e = int(np.floor(np.log10(v) + 1e-12))
    m = v / 10 ** e
    ms = f"{m:.2f}".rstrip("0").rstrip(".")
    return f"$10^{{{e}}}$" if ms == "1" else f"${ms}\\times10^{{{e}}}$"


def log_ticks(ax):
    """Labels of a logarithmic y-axis: the coarsest set of candidates that gives 2 to 6 labels inside the axis."""
    lo, hi = ax.get_ylim()
    span = np.log10(hi / lo)
    m = 0.03 * span
    inner = lambda v: np.log10(lo) + m <= np.log10(v) <= np.log10(hi) - m  # noqa: E731
    es = range(int(np.floor(np.log10(lo))) - 1, int(np.ceil(np.log10(hi))) + 1)
    sets = [(1,), (1, 2, 5), (1, 2, 3, 5, 7)] + [tuple(np.round(np.arange(1, 10, st), 2))
                                                  for st in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01)]
    ticks = []
    for subs in sets:
        ticks = sorted({round(mm * 10.0 ** e, 10) for e in es for mm in subs if inner(mm * 10.0 ** e)})
        if len(ticks) >= 2:
            break
    if len(ticks) > 6:
        ticks = ticks[:: int(np.ceil(len(ticks) / 6))]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _sci(v)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(lo, hi)


def style_axes(ax):
    for sp in ax.spines.values():
        sp.set_color("#8A9099")


# ---------------------------------------------------------------------------------------------------- CEC panels
def conv_panel(ax, exp, spec, main, suite, dim, func, title):
    for k, (alg, label, col, ls, mk) in enumerate(spec):
        C = exp.curves(suite, dim, func, alg)
        n = C.shape[1]
        m = np.nanmean(C, axis=0)
        x = 1000 * dim * np.arange(1, n + 1) / n
        ok = np.isfinite(m)
        is_main = label == main
        ax.plot(x[ok], m[ok], ls=ls, color=col, lw=1.35 if is_main else 0.75, zorder=10 if is_main else 3,
                marker=mk, ms=2.6 if is_main else 2.0, markevery=(k % 7, 7), mfc=col if is_main else "white",
                mec=col, mew=0.5)
    ax.set_yscale("log")
    ax.set_xlim(0, 1000 * dim)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax.yaxis.set_minor_formatter(NullFormatter())
    log_ticks(ax)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.get_offset_text().set_fontsize(6)
    ax.set_xticks(np.linspace(0, 1000 * dim, 5))
    ax.tick_params(labelsize=6.3, pad=1.5)
    ax.grid(True, which="major", color="#E3E6EA", lw=0.4, zorder=0)
    style_axes(ax)
    ax.set_title(title, fontsize=7.5, pad=2.5)


def box_panel(ax, exp, spec, main, suite, dim, func, title, xlabels):
    data = [exp.fitness(suite, dim, func, alg) for alg, *_ in spec]
    bp = ax.boxplot(data, widths=0.62, patch_artist=True, showfliers=True,
                    medianprops=dict(color="#1F1F1F", lw=0.9), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6),
                    flierprops=dict(marker="o", ms=1.8, mew=0.4, alpha=0.8))
    for k, (alg, label, col, ls, mk) in enumerate(spec):
        is_main = label == main
        bp["boxes"][k].set(facecolor=col, alpha=0.85 if is_main else 0.35, edgecolor=col, lw=1.0 if is_main else 0.6)
        for part in (bp["whiskers"][2 * k], bp["whiskers"][2 * k + 1], bp["caps"][2 * k], bp["caps"][2 * k + 1]):
            part.set(color=col)
        bp["fliers"][k].set(markeredgecolor=col, markerfacecolor="none")
        if is_main:
            bp["medians"][k].set(color="#1F1F1F", lw=1.1)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax.yaxis.set_minor_formatter(NullFormatter())
    log_ticks(ax)
    # a box thinner than 1.5% of the axis collapses to a line: mark its median with a short bar in the colour
    lo, hi = ax.get_ylim()
    for k, ((alg, label, col, ls, mk), v) in enumerate(zip(spec, data)):
        q1, q3 = np.percentile(v, [25, 75])
        if np.log10(q3 / q1) < 0.015 * np.log10(hi / lo):
            ax.plot([k + 1 - 0.31, k + 1 + 0.31], [np.median(v)] * 2, color=col, lw=2.2, solid_capstyle="butt",
                    zorder=4)
            ax.plot([k + 1 - 0.31, k + 1 + 0.31], [np.median(v)] * 2, color="#1F1F1F", lw=0.6, zorder=5)
    ax.set_xlim(0.4, len(spec) + 0.6)
    ax.set_xticks(range(1, len(spec) + 1))
    if xlabels:
        ax.set_xticklabels([label for _, label, *_ in spec], rotation=90, fontsize=6.0)
        for lab in ax.get_xticklabels():
            if lab.get_text() == main:
                lab.set_fontweight("bold")
    else:
        ax.set_xticklabels([])
    ax.tick_params(axis="y", labelsize=6.3, pad=1.5)
    ax.tick_params(axis="x", length=1.5, pad=1.5)
    ax.grid(True, axis="y", which="major", color="#E3E6EA", lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)
    ax.set_title(title, fontsize=7.5, pad=2.5)


def legend(fig, spec, main, ncol):
    H = [plt.Line2D([], [], ls=ls, color=col, lw=1.35 if label == main else 0.8, marker=mk,
                    ms=3 if label == main else 2.6, mfc=col if label == main else "white", mec=col, mew=0.5,
                    label=label) for _, label, col, ls, mk in spec]
    nrow = int(np.ceil(len(H) / ncol))
    H = [H[r * ncol + c] for c in range(ncol) for r in range(nrow) if r * ncol + c < len(H)]
    fig.legend(handles=H, loc="lower center", ncol=ncol, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.0),
               handlelength=2.6, columnspacing=1.2, handletextpad=0.4)


def main_grid(fig, axs, draw, ylabel, xlabel):
    for r, (suite, dim) in enumerate(CELLS):
        for c, func in enumerate(REP[suite]):
            draw(axs[r, c], suite, dim, func, f"{suite} F{func}", r == 3)
            if c == 0:
                axs[r, c].set_ylabel(ylabel, fontsize=7, labelpad=1)
            if r == 3 and xlabel:
                axs[r, c].set_xlabel(xlabel, fontsize=7, labelpad=1)
        bb = axs[r, 0].get_position()
        fig.text(0.012, (bb.y0 + bb.y1) / 2, f"{suite}, $D = {dim}$", rotation=90, ha="center", va="center",
                 fontsize=8, fontweight="bold")
    for c, cat in enumerate(CATS):
        bb = axs[0, c].get_position()
        fig.text((bb.x0 + bb.x1) / 2, 0.985, cat, ha="center", va="center", fontsize=8.5, fontweight="bold")


def fig_convergence(exp, out):
    fig, axs = plt.subplots(4, 4, figsize=(7.0, 7.6))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.955, bottom=0.095, wspace=0.3, hspace=0.42)
    main_grid(fig, axs, lambda ax, s, d, f, t, last: conv_panel(ax, exp, BENCH, "EpiRIME", s, d, f, t),
              "Mean best fitness", "FEs")
    legend(fig, BENCH, "EpiRIME", 6)
    save(fig, os.path.join(out, "fig_convergence"))


def fig_boxplot(exp, out):
    fig, axs = plt.subplots(4, 4, figsize=(7.0, 8.0))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.955, bottom=0.1, wspace=0.32, hspace=0.3)
    main_grid(fig, axs, lambda ax, s, d, f, t, last: box_panel(ax, exp, BENCH, "EpiRIME", s, d, f, t, last),
              "Best fitness", None)
    save(fig, os.path.join(out, "fig_boxplot"))


def supplementary(exp, spec, main, tag, ncol_leg, out):
    for suite, dim in CELLS:
        fl = FUNCS[suite]
        ncol = 5 if suite == "CEC2017" else 4
        nrow = int(np.ceil(len(fl) / ncol))
        # convergence
        H = 1.42 * nrow + 0.9
        fig, axs = plt.subplots(nrow, ncol, figsize=(7.0, H))
        fig.subplots_adjust(left=0.07, right=0.99, top=1 - 0.42 / H, bottom=0.62 / H + 0.02, wspace=0.3, hspace=0.5)
        for k, ax in enumerate(axs.flat):
            if k >= len(fl):
                ax.axis("off")
                continue
            conv_panel(ax, exp, spec, main, suite, dim, fl[k], f"F{fl[k]}")
            if k % ncol == 0:
                ax.set_ylabel("Mean best fitness", fontsize=6.5, labelpad=1)
            if k >= len(fl) - ncol:
                ax.set_xlabel("FEs", fontsize=6.5, labelpad=1)
        fig.suptitle(f"{suite}, $D = {dim}$", fontsize=9, fontweight="bold", y=1 - 0.08 / H)
        legend(fig, spec, main, ncol_leg)
        save(fig, os.path.join(out, f"figS_{tag}convergence_{suite}_D{dim}"))
        # box plots
        H = 1.55 * nrow + 1.1
        fig, axs = plt.subplots(nrow, ncol, figsize=(7.0, H))
        fig.subplots_adjust(left=0.07, right=0.99, top=1 - 0.42 / H, bottom=0.75 / H, wspace=0.32, hspace=0.35)
        for k, ax in enumerate(axs.flat):
            if k >= len(fl):
                ax.axis("off")
                continue
            box_panel(ax, exp, spec, main, suite, dim, fl[k], f"F{fl[k]}", k + ncol >= len(fl))
            if k % ncol == 0:
                ax.set_ylabel("Best fitness", fontsize=6.5, labelpad=1)
        fig.suptitle(f"{suite}, $D = {dim}$", fontsize=9, fontweight="bold", y=1 - 0.08 / H)
        save(fig, os.path.join(out, f"figS_{tag}boxplot_{suite}_D{dim}"))


# ---------------------------------------------------------------------------------------------------- rank figures
def rank_axes(ax, ylabel):
    ax.set_ylabel(ylabel, fontsize=7.5)
    ax.tick_params(labelsize=6.5, pad=1.5)
    style_axes(ax)


def fig_ablation(exp, out):
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.arange(len(ABL))
    res = {}
    for c, (suite, dim) in enumerate(CELLS):
        T = [[exp.fitness(suite, dim, f, alg).mean() for alg, *_ in ABL] for f in FUNCS[suite]]
        r = res[(suite, dim)] = mean_ranks(T)
        w = 0.19
        ax.bar(x + (c - 1.5) * w, r, width=w, color=CELL_COL[c], edgecolor="white", lw=0.3,
               label=f"{suite}, $D = {dim}$", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label, *_ in ABL], fontsize=6.8)
    for lab in ax.get_xticklabels():
        if lab.get_text() == "EpiRIME":
            lab.set_fontweight("bold")
            lab.set_color("#C0161E")
    ax.set_ylim(0, 4.2)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_xlim(-0.55, len(ABL) - 0.45)
    rank_axes(ax, "Mean Friedman rank")
    ax.tick_params(axis="x", length=0, pad=3)
    ax.grid(True, axis="y", color="#E3E6EA", lw=0.4, zorder=0)
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", ncol=2, handlelength=1.2, columnspacing=1.0,
              bbox_to_anchor=(0.5, -0.12))
    save(fig, os.path.join(out, "fig_ablation"))
    return res


def fig_sensitivity(exp, out):
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    res = {}
    for c, (suite, dim) in enumerate(CELLS):
        T = [[exp.fitness(suite, dim, f, alg).mean() for alg, *_ in SENS] for f in FUNCS[suite]]
        r = res[(suite, dim)] = mean_ranks(T)
        off = (c - 1.5) * 0.004                       # small horizontal shift to avoid overlapping markers
        ax.plot(np.array(RHOS) + off, r, color=CELL_COL[c], lw=1.0, marker="osD^"[c], ms=3.2, mfc="white", mew=0.8,
                label=f"{suite}, $D = {dim}$", zorder=4 + c)
    ax.axvline(0.3, color="#C0161E", lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(0.307, 1.2, r"default $\rho = 0.3$", color="#C0161E", fontsize=6.5, va="bottom", ha="left")
    ax.set_xlabel(r"Nucleus-pool ratio $\rho$", fontsize=7.5)
    ax.set_xticks(RHOS)
    ax.set_xticklabels([f"{r:g}" for r in RHOS], fontsize=6.3, rotation=0)
    ax.set_ylim(0.8, 10.2)
    ax.set_yticks(range(1, 11))
    rank_axes(ax, "Mean Friedman rank")
    ax.grid(True, color="#E3E6EA", lw=0.4, zorder=0)
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", ncol=2, handlelength=2.2, columnspacing=1.0,
              bbox_to_anchor=(0.5, -0.17))
    save(fig, os.path.join(out, "fig_sensitivity"))
    return res


def fig_runtime(folder, out):
    cpu = defaultdict(list)
    with open(os.path.join(folder, "cpu_time.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            cpu[(r["suite"], int(r["dim"]), r["algorithm"])].append(float(r["cpu_seconds"]))
    names = [label for _, label, *_ in BENCH]
    k = len(names)
    w = 0.19
    tab = {}
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    for c, (suite, dim) in enumerate(CELLS):
        vals = tab[(suite, dim)] = [np.mean(cpu[(suite, dim, alg)]) for alg, *_ in BENCH]
        ax.bar(np.arange(k) + (c - 1.5) * w, vals, width=w, color=CELL_COL[c], edgecolor="white", lw=0.3,
               label=f"{suite}, $D = {dim}$", zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels(names, fontsize=7, rotation=30, ha="right", rotation_mode="anchor")
    for lab in ax.get_xticklabels():
        if lab.get_text() == "EpiRIME":
            lab.set_fontweight("bold")
            lab.set_color("#C0161E")
    ax.set_ylabel("CPU time per run (s)", fontsize=7.5)
    ax.set_xlim(-0.6, k - 0.4)
    ax.tick_params(axis="y", labelsize=6.5, pad=1.5)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.grid(True, axis="y", which="major", color="#E3E6EA", lw=0.4, zorder=0)
    ax.axvspan(k - 1.5, k - 0.5, color="#C0161E", alpha=0.06, zorder=0)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=6.8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.2,
              columnspacing=1.5)
    save(fig, os.path.join(out, "fig_runtime"))
    return tab


# ---------------------------------------------------------------------------------------------------- gene selection
# the gene selection figures use the default (slightly longer and thicker) minor ticks
FS_RC = {"xtick.minor.size": 2.0, "xtick.minor.width": 0.6, "ytick.minor.size": 2.0, "ytick.minor.width": 0.6}


def load_fs(folder):
    R = defaultdict(dict)
    with open(os.path.join(folder, "runs.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            R[(r["algorithm"], r["dataset"])][int(r["run"])] = (int(r["n_genes"]), float(r["test_balanced_error"]))
    return R


@mpl.rc_context(FS_RC)
def fig_fs_tradeoff(R, out):
    fig, axs = plt.subplots(3, 3, figsize=(7.0, 6.2))
    for k, (d, name) in enumerate(FS_DS):
        ax = axs.flat[k]
        for i, a in enumerate(FS_ALGS):
            rows = list(R[(a, d)].values())
            assert len(rows) == 30, (a, d)
            n = np.mean([r[0] for r in rows])
            e = np.mean([r[1] for r in rows])
            c, main = FS_COL[a], a == "EpiRIME"
            ax.scatter(n, e, s=22 if main else 13, marker="o" if main else FS_MARK[i], facecolor=c if main else "none",
                       edgecolor=c, linewidth=0.7, zorder=3 if main else 2)
        ref = np.mean([GeneSelection(d, r).test_error(np.ones(2000)) for r in range(30)])     # all 2000 genes
        ax.axhline(ref, color="k", lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(0.7, 2500)
        ax.set_title(name, fontsize=7.5, pad=2.5)
        ax.tick_params(labelsize=6.3, pad=1.5)
        if k % 3 == 0:
            ax.set_ylabel("Mean test balanced error", fontsize=7, labelpad=1)
        if k >= 6:
            ax.set_xlabel("Mean number of selected genes", fontsize=7, labelpad=1)
    H = [plt.Line2D([], [], ls="", marker=FS_MARK[i], mfc="none", mec=FS_COL[a], mew=0.7, ms=3.8, label=a)
         for i, a in enumerate(FS_ALGS[:-1])]
    H.append(plt.Line2D([], [], ls="", marker="o", mfc=FS_COL["EpiRIME"], mec=FS_COL["EpiRIME"], mew=0.7,
                        ms=4.5, label="EpiRIME"))
    H.append(plt.Line2D([], [], color="k", lw=0.6, ls=(0, (3, 2)), label="All 2000 genes"))
    fig.legend(handles=H, loc="lower center", ncol=5, frameon=False, fontsize=6.6, bbox_to_anchor=(0.5, 0.0),
               handletextpad=0.2, columnspacing=0.9)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.115, wspace=0.25, hspace=0.33)
    fig.savefig(os.path.join(out, "fig_fs_tradeoff.pdf"), metadata=META)
    fig.savefig(os.path.join(out, "fig_fs_tradeoff.png"), dpi=600)
    plt.close(fig)


@mpl.rc_context(FS_RC)
def fig_fs_testbox(R, out):
    fig, axs = plt.subplots(3, 3, figsize=(7.0, 6.6))
    for k, (d, name) in enumerate(FS_DS):
        ax = axs.flat[k]
        data = [[R[(a, d)][r][1] for r in sorted(R[(a, d)])] for a in FS_ALGS]
        assert all(len(v) == 30 for v in data)
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=True,
                        medianprops=dict(color="#1F1F1F", lw=0.9), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6),
                        flierprops=dict(marker="o", ms=1.8, mew=0.4, alpha=0.8))
        for i, a in enumerate(FS_ALGS):
            c, main = FS_COL[a], a == "EpiRIME"
            bp["boxes"][i].set(facecolor=c, alpha=0.85 if main else 0.35, edgecolor=c, lw=1.0 if main else 0.6)
            for part in (bp["whiskers"][2 * i], bp["whiskers"][2 * i + 1], bp["caps"][2 * i], bp["caps"][2 * i + 1]):
                part.set(color=c)
            bp["fliers"][i].set(markeredgecolor=c, markerfacecolor="none")
            ax.scatter(i + 1, np.mean(data[i]), marker="D", s=9, color="white", edgecolor="#1F1F1F", lw=0.5, zorder=5)
        ax.set_title(name, fontsize=7.5, pad=2.5)
        ax.set_xticks(range(1, len(FS_ALGS) + 1))
        if k >= 6:
            ax.set_xticklabels(FS_ALGS, rotation=90, fontsize=6.3)
            for lab in ax.get_xticklabels():
                if lab.get_text() == "EpiRIME":
                    lab.set_fontweight("bold")
                    lab.set_color(FS_COL["EpiRIME"])
        else:
            ax.set_xticklabels([])
        ax.tick_params(axis="y", labelsize=6.3, pad=1.5)
        ax.tick_params(axis="x", length=1.5, pad=1.5)
        ax.grid(True, axis="y", color="#E3E6EA", lw=0.4, zorder=0)
        ax.set_axisbelow(True)
        if k % 3 == 0:
            ax.set_ylabel("Test balanced error", fontsize=7, labelpad=1)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.13, wspace=0.25, hspace=0.25)
    fig.savefig(os.path.join(out, "fig_fs_testbox.pdf"), metadata=META)
    fig.savefig(os.path.join(out, "fig_fs_testbox.png"), dpi=600)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True, help="folder of the results package (EpiRIME-results)")
    ap.add_argument("--out", default="figures", help="output folder")
    ap.add_argument("--only", nargs="*", default=None,
                    help="draw only some figures: main, supplementary, gene_selection")
    a = ap.parse_args()
    parts = set(a.only or ["main", "supplementary", "gene_selection"])
    os.makedirs(a.out, exist_ok=True)
    assert set(DATASETS) == {d for d, _ in FS_DS}
    bench = Experiment(os.path.join(a.results, "benchmark"))
    abl = Experiment(os.path.join(a.results, "ablation"))
    sens = Experiment(os.path.join(a.results, "sensitivity"))
    if "main" in parts:
        fig_convergence(bench, a.out)
        fig_boxplot(bench, a.out)
        r2 = lambda v: [round(float(x), 2) for x in v]  # noqa: E731
        for key, r in fig_ablation(abl, a.out).items():
            print("ablation, mean ranks", *key, dict(zip([lab for _, lab, *_ in ABL], r2(r))))
        for key, r in fig_sensitivity(sens, a.out).items():
            print("sensitivity, mean ranks", *key, dict(zip(RHOS, r2(r))))
        for key, v in fig_runtime(os.path.join(a.results, "benchmark"), a.out).items():
            print("CPU seconds per run", *key, dict(zip([lab for _, lab, *_ in BENCH], r2(v))))
    if "supplementary" in parts:
        supplementary(bench, BENCH, "EpiRIME", "", 6, a.out)
        supplementary(abl, ABL, "EpiRIME", "abl_", 4, a.out)
        supplementary(sens, SENS, r"$\rho = 0.3$", "sens_", 5, a.out)
    if "gene_selection" in parts:
        R = load_fs(os.path.join(a.results, "gene_selection"))
        fig_fs_tradeoff(R, a.out)
        fig_fs_testbox(R, a.out)
    print("figures written to", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
