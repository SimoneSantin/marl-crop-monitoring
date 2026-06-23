"""
generate_thesis_plots.py

Genera i grafici per il Capitolo 8 della tesi.
File JSON attesi in: results/<config_name>/seed_<seed>.json

Output:
    thesis_plots/
    ├── 8.1_summary/
    │   ├── summary_table.csv
    │   └── summary_table.tex
    ├── 8.2_ablation/
    │   ├── global_accuracy.png       NoBelief · Bayesian · GF · Oracle
    │   ├── unvisited_accuracy.png    NoBelief · Bayesian · GF · Oracle
    │   └── coverage.png              NoBelief · Bayesian · GF · Oracle
    ├── 8.3_lstm_analysis/
    │   ├── global_accuracy.png       GF · LSTM · LSTM+GF · Oracle
    │   ├── unvisited_accuracy.png    GF · LSTM · LSTM+GF · Oracle
    │   └── coverage.png              GF · LSTM · LSTM+GF · Oracle
    ├── 8.4_tradeoff_behavior/
    │   ├── collisions_barplot.png    NoBelief · Bayesian · GF · LSTM · LSTM+GF
    │   ├── episodes_to_target.png    NoBelief · Bayesian · GF · LSTM · LSTM+GF
    │   └── accuracy_vs_coverage.png  NoBelief · Bayesian · GF · LSTM · LSTM+GF
    └── appendice/
        ├── tabella_per_seed.tex
        ├── visited_accuracy_ablation.png
        ├── visited_accuracy_lstm.png
        └── accuracy_vs_coverage_seed.png

Uso:
    python generate_thesis_plots.py
    python generate_thesis_plots.py --results_dir results --output_dir thesis_plots
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAZIONE GLOBALE
# ─────────────────────────────────────────────────────────────────────────────

SEEDS = [42, 123, 456]

CONFIGS = {
    "2_mappo_no_belief": {"label": "No Belief",      "color": "#E07B39", "ls": "-"},
    "3_mappo_bayesian":  {"label": "Bayesian",        "color": "#D4A017", "ls": "-"},
    "4_mappo_bayes_gf":  {"label": "Bayesian + GF",   "color": "#2E75B6", "ls": "-"},
    "5_mappo_lstm":      {"label": "LSTM puro",        "color": "#7030A0", "ls": "-"},
    "6_mappo_gf_lstm":   {"label": "LSTM + GF",        "color": "#00B050", "ls": "-"},
    "7_oracle_conf":     {"label": "Oracle",           "color": "#C00000", "ls": "--"},
}

# Gruppi per sezione — GF e Oracle si ripetono in 8.2 e 8.3: è corretto,
# in 8.2 GF è il punto di arrivo dell'ablation, in 8.3 è la baseline di partenza.
GROUP_82 = ["2_mappo_no_belief", "3_mappo_bayesian", "4_mappo_bayes_gf", "7_oracle_conf"]
GROUP_83 = ["4_mappo_bayes_gf",  "5_mappo_lstm",     "6_mappo_gf_lstm",  "7_oracle_conf"]
GROUP_84 = ["2_mappo_no_belief", "3_mappo_bayesian",  "4_mappo_bayes_gf",
            "5_mappo_lstm",      "6_mappo_gf_lstm"]

# Configurazioni per appendice (escluso No Belief perché accuracy = prior)
GROUP_APP = ["3_mappo_bayesian", "4_mappo_bayes_gf",
             "5_mappo_lstm",     "6_mappo_gf_lstm", "7_oracle_conf"]

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   9.5,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

# ─────────────────────────────────────────────────────────────────────────────
# CARICAMENTO DATI
# ─────────────────────────────────────────────────────────────────────────────

def load_data(results_dir):
    data = {}
    for config_name in CONFIGS:
        data[config_name] = {}
        for seed in SEEDS:
            path = os.path.join(results_dir, config_name, f"seed_{seed}.json")
            if os.path.exists(path):
                with open(path) as f:
                    data[config_name][seed] = json.load(f)
            else:
                print(f"  [MANCANTE] {path}")
                data[config_name][seed] = None
    return data


def get_mean_history(data, config_name, key):
    """Media di una history sui seed disponibili."""
    histories = [
        data[config_name][s][key]
        for s in SEEDS
        if data[config_name][s] is not None
        and data[config_name][s].get(key)
    ]
    if not histories:
        return None, None
    min_len = min(len(h) for h in histories)
    arr = np.array([h[:min_len] for h in histories])
    return np.arange(min_len), arr.mean(axis=0)


def get_scalar(data, config_name, key):
    """Lista dei valori scalari per i 3 seed."""
    return [
        data[config_name][s][key]
        if data[config_name][s] is not None and key in data[config_name][s]
        else None
        for s in SEEDS
    ]


def get_final_from_history(data, config_name, key):
    """Valore finale (ultimo elemento) di una history, per ogni seed."""
    vals = []
    for s in SEEDS:
        d = data[config_name][s]
        if d is not None and d.get(key):
            vals.append(d[key][-1])
        else:
            vals.append(None)
    return vals


def moving_average(arr, window=30):
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def makedirs(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# GRAFICI TEMPORALI
# ─────────────────────────────────────────────────────────────────────────────

def plot_temporal(data, group, metric_key, title, ylabel, output_path,
                  window=30, focus_labels=None, thin_labels=None):
    """Una curva per configurazione (media sui seed, smoothed).

    Modifiche di leggibilità:
    - figura riportata alla dimensione originale;
    - linee complessivamente più sottili;
    - alcune curve di riferimento ulteriormente assottigliate;
    - curve secondarie leggermente trasparenti;
    - etichette dirette a fine curva;
    - legenda fuori dal grafico.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    if focus_labels is None:
        focus_labels = {"Oracle"}
    else:
        focus_labels = set(focus_labels)

    if thin_labels is None:
        thin_labels = set()
    else:
        thin_labels = set(thin_labels)

    line_ends = []
    max_x = 0

    for config_name in group:
        cfg = CONFIGS[config_name]
        eps, mean = get_mean_history(data, config_name, metric_key)
        if eps is None:
            print(f"  [SKIP] {config_name} — {metric_key} non disponibile")
            continue

        sm = moving_average(mean, window)
        x = eps[:len(sm)]
        if len(x) == 0:
            continue

        label = cfg["label"]
        is_focus = label in focus_labels
        is_thin = label in thin_labels

        # Linee più sottili rispetto alla versione precedente:
        # - focus: leggermente più evidente, ma non troppo spesso;
        # - standard: più leggero;
        # - thin: curve di contesto/riferimento ulteriormente ridotte.
        if is_thin:
            linewidth = 1.15
        elif is_focus:
            linewidth = 2.05
        else:
            linewidth = 1.55

        alpha = 0.85 if is_thin else (1.0 if is_focus else 0.78)
        zorder = 4 if is_focus and not is_thin else 3

        ax.plot(
            x,
            sm,
            color=cfg["color"],
            linestyle=cfg["ls"],
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            zorder=zorder,
        )

        max_x = max(max_x, int(x[-1]))
        line_ends.append({
            "label": label,
            "x": float(x[-1]),
            "y": float(sm[-1]),
            "color": cfg["color"],
            "alpha": alpha,
            "is_focus": is_focus,
        })

    ax.set_title(title)
    ax.set_xlabel("Episodio")
    ax.set_ylabel(ylabel)

    # Spazio a destra per le etichette dirette.
    if line_ends:
        ax.set_xlim(0, max_x + 55)

        # Evita sovrapposizioni tra etichette finali troppo vicine.
        ymin, ymax = ax.get_ylim()
        min_sep = (ymax - ymin) * 0.035
        ordered = sorted(line_ends, key=lambda item: item["y"])
        adjusted = []
        last_y = None
        for item in ordered:
            y = item["y"] if last_y is None else max(item["y"], last_y + min_sep)
            adjusted.append((item, y))
            last_y = y

        # Se le etichette superano il limite superiore, le riporta dentro il grafico.
        overflow = adjusted[-1][1] - ymax if adjusted else 0
        if overflow > 0:
            adjusted = [(item, y - overflow) for item, y in adjusted]

        for item, y_text in adjusted:
            ax.text(
                max_x + 8,
                y_text,
                item["label"],
                color=item["color"],
                alpha=item["alpha"],
                fontsize=8.5,
                fontweight="bold" if item["is_focus"] and item["alpha"] >= 0.99 else "normal",
                va="center",
                clip_on=False,
            )

    # Legenda fuori dal grafico per non coprire le curve.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=min(len(group), 4),
        framealpha=0.85,
    )

    fig.tight_layout()
    makedirs(output_path)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Salvato: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# BAR PLOT — collisioni
# ─────────────────────────────────────────────────────────────────────────────

def plot_collisions_barplot(data, group, output_path):
    """Bar plot delle collisioni medie con punti seed visibili ed etichettati."""
    labels, means, colors, seed_vals = [], [], [], []

    for config_name in group:
        cfg = CONFIGS[config_name]
        vals = []
        for seed in SEEDS:
            d = data[config_name][seed]
            if d is not None and "mean_collisions" in d:
                vals.append((seed, d["mean_collisions"]))
        if not vals:
            continue
        labels.append(cfg["label"])
        means.append(np.mean([v for _, v in vals]))
        colors.append(cfg["color"])
        seed_vals.append(vals)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x, means, color=colors, width=0.55, alpha=0.85, zorder=2)

    for i, vals in enumerate(seed_vals):
        jitter = np.linspace(-0.1, 0.1, len(vals))
        for j, (seed, v) in enumerate(vals):
            px = x[i] + jitter[j]
            ax.scatter(px, v, color="black", s=28, zorder=4, alpha=0.8)
            ax.annotate(
                str(seed),
                xy=(px, v),
                xytext=(3, 4),
                textcoords="offset points",
                fontsize=7.5,
                color="black",
                zorder=5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Collisioni medie per episodio")
    ax.set_title("Collisioni medie per configurazione")
    fig.tight_layout()
    makedirs(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvato: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# BAR PLOT — episodi a target accuracy
# ─────────────────────────────────────────────────────────────────────────────

def plot_episodes_to_target(data, group, output_path):
    """Bar plot degli episodi per raggiungere soglie 0.75 e 0.80."""
    max_ep = 500
    labels, vals_075, vals_080, colors = [], [], [], []

    for config_name in group:
        cfg = CONFIGS[config_name]
        ep075 = [data[config_name][s]["episodes_to_075"]
                 for s in SEEDS
                 if data[config_name][s] is not None
                 and "episodes_to_075" in data[config_name][s]]
        ep080 = [data[config_name][s]["episodes_to_080"]
                 for s in SEEDS
                 if data[config_name][s] is not None
                 and "episodes_to_080" in data[config_name][s]]
        if not ep075:
            continue
        labels.append(cfg["label"])
        vals_075.append(np.mean(ep075))
        vals_080.append(np.mean(ep080) if ep080 else max_ep)
        colors.append(cfg["color"])

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w/2, vals_075, width=w, color=colors, alpha=0.9,
           label="Target 0.75", zorder=2)
    ax.bar(x + w/2, vals_080, width=w, color=colors, alpha=0.55,
           label="Target 0.80", zorder=2, hatch="//")

    for i, (v75, v80) in enumerate(zip(vals_075, vals_080)):
        if v75 >= max_ep:
            ax.text(x[i] - w/2, max_ep + 8, "n.r.",
                    ha="center", fontsize=8, color="gray")
        if v80 >= max_ep:
            ax.text(x[i] + w/2, max_ep + 8, "n.r.",
                    ha="center", fontsize=8, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Episodio (media sui 3 seed)")
    ax.set_title("Episodi per raggiungere il target di accuratezza")
    ax.set_ylim(0, max_ep * 1.15)
    # Legenda fuori dal grafico.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        framealpha=0.85,
    )
    fig.tight_layout()
    makedirs(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvato: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SCATTER — accuracy vs coverage (versione 8.4, con etichette seed)
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_vs_coverage(data, group, output_path):
    """Scatter: punti seed etichettati, senza stella della media."""
    fig, ax = plt.subplots(figsize=(7, 5))
    legend_handles = []

    for config_name in group:
        cfg = CONFIGS[config_name]
        has_points = False

        for seed in SEEDS:
            d = data[config_name][seed]
            cov = d["coverage_history"][-1] if d and d.get("coverage_history") else None
            acc = d["accuracy_visited_history"][-1] if d and d.get("accuracy_visited_history") else None

            if cov is None or acc is None:
                continue

            has_points = True
            ax.scatter(cov, acc,
                       color=cfg["color"], s=55, alpha=0.75, zorder=3)
            ax.annotate(
                str(seed),
                xy=(cov, acc),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.5,
                color=cfg["color"],
                zorder=4,
            )

        if has_points:
            legend_handles.append(
                mpatches.Patch(color=cfg["color"], label=cfg["label"]))

    ax.set_xlabel("Coverage finale")
    ax.set_ylabel("Visited accuracy finale")
    ax.set_title("Trade-off accuratezza / copertura")

    # Legenda fuori dal grafico.
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8.5,
        framealpha=0.85,
    )

    fig.tight_layout()
    makedirs(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvato: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# TABELLE LATEX
# ─────────────────────────────────────────────────────────────────────────────

def export_summary_table_csv(data, output_path):
    import csv
    rows = []
    for config_name, cfg in CONFIGS.items():
        vis  = [v for v in get_final_from_history(
                    data, config_name, "accuracy_visited_history")
                if v is not None]
        unv  = [v for v in get_final_from_history(
                    data, config_name, "accuracy_unvisited_history")
                if v is not None]
        cov  = [v for v in get_final_from_history(
                    data, config_name, "coverage_history")
                if v is not None]
        rows.append({
            "Configurazione":   cfg["label"],
            "Visited accuracy": f"{np.mean(vis):.3f}" if vis else "n/a",
            "Unvisited accuracy": f"{np.mean(unv):.3f}" if unv else "n/a",
            "Coverage":         f"{np.mean(cov):.3f}" if cov else "n/a",
        })
    makedirs(output_path)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Salvato: {output_path}")


def export_summary_table_latex(data, output_path):
    def fmt(vals):
        if not vals:
            return "—"
        v = np.mean(vals)
        if v < 0.12:
            return f"\\textcolor{{gray}}{{{v:.3f}}}"
        return f"{v:.3f}"

    lines = [
        "% Tabella 8.1 — Risultati riassuntivi (media sui 3 seed)",
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Confronto delle configurazioni sperimentali. "
        "Valori medi sui tre semi (42, 123, 456).}",
        "\\label{tab:results_summary}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Configurazione} & \\textbf{Visited} "
        "& \\textbf{Unvisited} & \\textbf{Coverage} \\\\",
        "\\midrule",
    ]
    for config_name, cfg in CONFIGS.items():
        vis = [v for v in get_final_from_history(
                   data, config_name, "accuracy_visited_history")
               if v is not None]
        unv = [v for v in get_final_from_history(
                   data, config_name, "accuracy_unvisited_history")
               if v is not None]
        cov = [v for v in get_final_from_history(
                   data, config_name, "coverage_history")
               if v is not None]
        label = cfg["label"]
        if "Oracle" in label:
            lines.append("\\midrule")
            lines.append(f"\\textit{{{label}}} & \\textit{{{fmt(vis)}}} "
                         f"& \\textit{{{fmt(unv)}}} & \\textit{{{fmt(cov)}}} \\\\")
        else:
            lines.append(f"{label} & {fmt(vis)} & {fmt(unv)} & {fmt(cov)} \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    makedirs(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Salvato: {output_path}")


def export_seed_table_latex(data, output_path):
    lines = [
        "% Tabella appendice — Risultati per seed",
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        "\\caption{Risultati per seed. Valori finali all'episodio 500.}",
        "\\label{tab:results_per_seed}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "\\textbf{Configurazione} & \\textbf{Seed} "
        "& \\textbf{Visited} & \\textbf{Unvisited} & \\textbf{Coverage} \\\\",
        "\\midrule",
    ]
    for config_name in GROUP_APP:
        cfg = CONFIGS[config_name]
        first = True
        for seed in SEEDS:
            d = data[config_name][seed]
            vis  = f"{d['accuracy_visited_history'][-1]:.3f}"   \
                   if d and d.get("accuracy_visited_history")   else "—"
            unv  = f"{d['accuracy_unvisited_history'][-1]:.3f}" \
                   if d and d.get("accuracy_unvisited_history") else "—"
            cov  = f"{d['coverage_history'][-1]:.3f}"           \
                   if d and d.get("coverage_history")           else "—"
            label_col = cfg["label"] if first else ""
            lines.append(f"{label_col} & {seed} & {vis} & {unv} & {cov} \\\\")
            first = False
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table}"]
    makedirs(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Salvato: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SCATTER APPENDICE — con etichette seed
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_vs_coverage_seed(data, group, output_path):
    """Scatter con etichetta seed su ogni punto, senza stella della media."""
    fig, ax = plt.subplots(figsize=(8, 6))
    legend_handles = []

    for config_name in group:
        cfg = CONFIGS[config_name]
        has_points = False

        for seed in SEEDS:
            d = data[config_name][seed]
            cov = d["coverage_history"][-1] if d and d.get("coverage_history") else None
            acc = d["accuracy_visited_history"][-1] if d and d.get("accuracy_visited_history") else None

            if cov is None or acc is None:
                continue

            has_points = True
            ax.scatter(cov, acc, color=cfg["color"], s=55,
                       alpha=0.8, zorder=3)
            ax.annotate(str(seed), xy=(cov, acc), xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=7.5, color=cfg["color"], zorder=4)

        if has_points:
            legend_handles.append(
                mpatches.Patch(color=cfg["color"], label=cfg["label"]))

    ax.set_xlabel("Coverage finale")
    ax.set_ylabel("Visited accuracy finale")
    ax.set_title("Trade-off accuratezza / copertura — dettaglio per seed")

    # Legenda fuori dal grafico.
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8.5,
        framealpha=0.85,
    )

    fig.tight_layout()
    makedirs(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvato: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output_dir",  default="thesis_plots")
    args = parser.parse_args()
    R, O = args.results_dir, args.output_dir

    print(f"\nCaricamento dati da: {R}")
    data = load_data(R)
    # ── 8.1 Tabella riassuntiva ──────────────────────────────────────────────
    print("\n── 8.1 Tabella riassuntiva ─────────────────────────────")
    d81 = os.path.join(O, "8.1_summary")
    export_summary_table_csv(data,   os.path.join(d81, "summary_table.csv"))
    export_summary_table_latex(data, os.path.join(d81, "summary_table.tex"))

    # ── 8.2 Ablation study ──────────────────────────────────────────────────
    # NoBelief · Bayesian · GF · Oracle
    print("\n── 8.2 Ablation study ──────────────────────────────────")
    d82 = os.path.join(O, "8.2_ablation")
    plot_temporal(data, GROUP_82, "accuracy_history",
                  "Global accuracy — Ablation study",
                  "Global accuracy (media 3 seed)",
                  os.path.join(d82, "global_accuracy.png"),
                  focus_labels=["Bayesian", "Bayesian + GF"],
                  thin_labels=["No Belief", "Oracle"])
    plot_temporal(data, GROUP_82, "accuracy_unvisited_history",
                  "Unvisited accuracy — Ablation study",
                  "Unvisited accuracy (media 3 seed)",
                  os.path.join(d82, "unvisited_accuracy.png"),
                  focus_labels=["Bayesian", "Bayesian + GF"],
                  thin_labels=["No Belief", "Oracle"])
    plot_temporal(data, GROUP_82, "coverage_history",
                  "Coverage — Ablation study",
                  "Coverage (media 3 seed)",
                  os.path.join(d82, "coverage.png"),
                  window=45,
                  focus_labels=["Bayesian", "Bayesian + GF"],
                  thin_labels=["No Belief", "Oracle"])

    # ── 8.3 Analisi LSTM ────────────────────────────────────────────────────
    # GF · LSTM · LSTM+GF · Oracle
    print("\n── 8.3 Analisi LSTM ────────────────────────────────────")
    d83 = os.path.join(O, "8.3_lstm_analysis")
    plot_temporal(data, GROUP_83, "accuracy_history",
                  "Global accuracy — Contributo LSTM",
                  "Global accuracy (media 3 seed)",
                  os.path.join(d83, "global_accuracy.png"),
                  focus_labels=["LSTM + GF"],
                  thin_labels=["Bayesian + GF", "Oracle"])
    plot_temporal(data, GROUP_83, "accuracy_unvisited_history",
                  "Unvisited accuracy — Contributo LSTM",
                  "Unvisited accuracy (media 3 seed)",
                  os.path.join(d83, "unvisited_accuracy.png"),
                  focus_labels=["LSTM + GF"],
                  thin_labels=["Bayesian + GF", "Oracle"])
    plot_temporal(data, GROUP_83, "coverage_history",
                  "Coverage — Contributo LSTM",
                  "Coverage (media 3 seed)",
                  os.path.join(d83, "coverage.png"),
                  window=45,
                  focus_labels=["LSTM + GF"],
                  thin_labels=["Bayesian + GF", "Oracle"])

    # ── 8.4 Trade-off e comportamento ───────────────────────────────────────
    # NoBelief · Bayesian · GF · LSTM · LSTM+GF
    print("\n── 8.4 Trade-off e comportamento ───────────────────────")
    d84 = os.path.join(O, "8.4_tradeoff_behavior")
    plot_collisions_barplot(data, GROUP_84,
                            os.path.join(d84, "collisions_barplot.png"))
    plot_episodes_to_target(data, GROUP_84,
                            os.path.join(d84, "episodes_to_target.png"))
    plot_accuracy_vs_coverage(data, GROUP_84,
                              os.path.join(d84, "accuracy_vs_coverage.png"))

    # ── Appendice ────────────────────────────────────────────────────────────
    print("\n── Appendice ────────────────────────────────────────────")
    dapp = os.path.join(O, "appendice")
    export_seed_table_latex(data, os.path.join(dapp, "tabella_per_seed.tex"))
    plot_temporal(data, GROUP_82, "accuracy_visited_history",
                  "Visited accuracy — Ablation study",
                  "Visited accuracy (media 3 seed)",
                  os.path.join(dapp, "visited_accuracy_ablation.png"),
                  focus_labels=["Bayesian", "Bayesian + GF", "Oracle"])
    plot_temporal(data, GROUP_83, "accuracy_visited_history",
                  "Visited accuracy — Contributo LSTM",
                  "Visited accuracy (media 3 seed)",
                  os.path.join(dapp, "visited_accuracy_lstm.png"),
                  focus_labels=["LSTM + GF", "Oracle"])
    plot_accuracy_vs_coverage_seed(data, GROUP_APP,
                                   os.path.join(dapp, "accuracy_vs_coverage_seed.png"))

    print(f"\n✓ Tutti i grafici salvati in: {O}/")


if __name__ == "__main__":
    main()
