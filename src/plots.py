"""Rezultātu grafiki (matplotlib, PNG)."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"     # 1. kategoriskā krāsa
ORANGE = "#eb6834"   # 2. kategoriskā krāsa
MUTED = "#a3a29c"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "text.color": TEXT,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.titlecolor": TEXT,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "font.size": 9, "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
})


def _label(s):
    return f"{s['m']}/{s['n']}"


def plot_gap(summaries, rows, path):
    """Novirze % no etalona katrā palaišanā + vidējais."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, s in enumerate(summaries):
        g = np.array([r["gap_pct"] for r in rows if r["instance"] == s["instance"]])
        jitter = np.linspace(-0.18, 0.18, len(g)) if len(g) > 1 else np.zeros(1)
        ax.scatter(i + jitter, g, s=22, color=BLUE, alpha=0.55, edgecolor=SURFACE, linewidth=1,
                   label="viena palaišana" if i == 0 else None, zorder=3)
        ax.hlines(g.mean(), i - 0.3, i + 0.3, color=TEXT, linewidth=2,
                  label="vidējais" if i == 0 else None, zorder=4)
        ax.annotate(f"{g.mean():.2f}%", (i + 0.32, g.mean()), va="center", fontsize=8, color=TEXT_2)
    ax.set_xticks(range(len(summaries)), [_label(s) for s in summaries])
    ax.set_xlim(-0.5, len(summaries) - 0.3)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("skolēni / skolas")
    ax.set_ylabel("novirze no optimuma, %")
    ax.set_title("Simulētās rūdīšanas novirze no precīzā optimuma")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_time(summaries, path):
    """Vidējais SA laiks un precīzā algoritma laiks atkarībā no izmēra."""
    m = [s["m"] for s in summaries]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    sa = [s["sa_time_mean_s"] for s in summaries]
    ax.plot(m, sa, color=BLUE, marker="o", markersize=6, markeredgecolor=SURFACE, label="simulētā rūdīšana (vid.)")
    ex = [(s["m"], s["exact_time_s"]) for s in summaries if s["exact_time_s"] is not None]
    if ex:
        ax.plot(*zip(*ex), color=ORANGE, marker="s", markersize=6, markeredgecolor=SURFACE,
                label="precīzais (min-cost flow)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(m, [_label(s) for s in summaries])
    ax.minorticks_off()
    ax.set_xlim(m[0] * 0.8, m[-1] * 1.25)
    ax.set_xlabel("skolēni / skolas (log)")
    ax.set_ylabel("laiks, s (log)")
    ax.set_title("Izpildes laiks")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_convergence(summaries, histories, path):
    """Konverģence (sēkla 0): pašreizējā un labākā risinājuma novirze pret iterāciju."""
    k = len(summaries)
    fig, axes = plt.subplots(1, k, figsize=(3.0 * k, 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, summaries):
        ref, hist = histories[s["instance"]]
        h = np.array(hist, dtype=float)
        it, cur, best = h[:, 0], h[:, 2], h[:, 3]
        ax.plot(it, 100 * (cur - ref) / ref, color=MUTED, linewidth=1.5, label="pašreizējais")
        ax.plot(it, 100 * (best - ref) / ref, color=BLUE, label="labākais")
        ax.set_yscale("symlog", linthresh=1)
        ax.set_title(_label(s), fontsize=10)
        ax.set_xlabel("iterācija")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 3))
    axes[0].set_ylabel("novirze no optimuma, % (symlog)")
    axes[0].legend(loc="upper right")
    fig.suptitle("Konverģence (sēkla 0)", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_alpha(sweep, path):
    """Alfa ietekme: novirze un laiks divos atsevišķos paneļos (bez divām y asīm)."""
    alphas = sorted({r["alpha"] for r in sweep})
    g = [np.mean([r["gap_pct"] for r in sweep if r["alpha"] == a]) for a in alphas]
    t = [np.mean([r["time_s"] for r in sweep if r["alpha"] == a]) for a in alphas]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.2))
    x = range(len(alphas))
    for ax, y, lab, fmt in ((a1, g, "vid. novirze, %", "{:.2f}%"), (a2, t, "vid. laiks, s", "{:.1f} s")):
        ax.bar(x, y, width=0.55, color=BLUE)
        for xi, yi in zip(x, y):
            ax.annotate(fmt.format(yi), (xi, yi), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, color=TEXT_2)
        ax.set_xticks(list(x), [f"{a:.2f}" for a in alphas])
        ax.set_xlabel("alfa")
        ax.set_ylabel(lab)
        ax.grid(axis="x", visible=False)
    a1.set_title("Novirze no optimuma")
    a2.set_title("Laiks")
    fig.suptitle(f"Dzesēšanas koeficienta ietekme ({sweep[0]['instance']}, 5 sēklas)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_plots(summaries, rows, histories, sweep, out):
    paths = [os.path.join(out, "gap.png"), os.path.join(out, "time.png"),
             os.path.join(out, "convergence.png")]
    plot_gap(summaries, rows, paths[0])
    plot_time(summaries, paths[1])
    plot_convergence(summaries, histories, paths[2])
    if sweep:
        paths.append(os.path.join(out, "alpha.png"))
        plot_alpha(sweep, paths[-1])
    return paths
