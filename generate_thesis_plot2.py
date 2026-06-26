"""
generate_thesis_plots.py

Genera i grafici per il Capitolo 8 della tesi.
File JSON attesi in: results/<config_name>/seed_<seed>.json

Differenze rispetto alla versione precedente:
  - Curve temporali: banda di varianza (±SEM di default) disegnata SOLO sulle
    curve indicate da `band_labels` (di norma le curve "focus"), per evitare il
    sovraffollamento con 4 linee. Le curve di riferimento (No Belief, Oracle)
    non hanno banda.
  - 8.4 trade-off: rimossi i punti per seed. La variabilità è mostrata con
    barre di errore (±1 std):
      * collisions_barplot  -> barre di errore verticali
      * episodes_to_target  -> barre di errore verticali
      * accuracy_vs_coverage -> un punto medio per configurazione con barre di
        errore su entrambi gli assi (x = coverage, y = visited)
  - Appendice: lo scatter per seed resta con i punti seed (serve proprio a
    mostrare il dettaglio); la tabella per seed resta invariata.

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

GROUP_82 = ["2_mappo_no_belief", "3_mappo_bayesian", "4_mappo_bayes_gf", "7_oracle_conf"]
GROUP_83 = ["4_mappo_bayes_gf",  "5_mappo_lstm",     "6_mappo_gf_lstm",  "7_oracle_conf"]
GROUP_84 = ["2_mappo_no_belief", "3_mappo_bayesian",  "4_mappo_bayes_gf",
            "5_mappo_lstm",      "6_mappo_gf_lstm"]
GROUP_APP = ["3_mappo_bayesian", "4_mappo_bayes_gf",
             "5_mappo_lstm",     "6_mappo_gf_lstm", "7_oracle_conf"]

# Tipo di banda di varianza per le curve temporali:
#   "sem"  -> ±1 errore standard (std/sqrt(n))  [consigliato con 3 seed]
#   "std"  -> ±1 deviazione standard
#   "half" -> ±0.5 deviazione standard
BAND_KIND = "sem"

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
    """Media di una history sui seed disponibili (senza banda)."""
    histories = [
        data[config_name][s][key]
        for s in SEEDS
        if data[config_name][s] is not None and data[config_name][s].get(key)
    ]
    if not histories:
        return None, None
    min_len = min(len(h) for h in histories)
    arr = np.array([h[:min_len] for h in histories])
    return np.arange(min_len), arr.mean(axis=0)


def get_mean_band_history(data, config_name, key, band=BAND_KIND):
    """Media + semiampiezza della banda di varianza, puntuale sui seed.

    Ritorna (episodi, media, half) dove la banda da disegnare è
    [media - half, media + half].
    """
    histories = [
        data[config_name][s][key]
        for s in SEEDS
        if data[config_name][s] is not None and data[config_name][s].get(key)
    ]
    if not histories:
        return None, None, None
    min_len = min(len(h) for h in histories)
    arr = np.array([h[:min_len] for h in histories])
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
    n = arr.shape[0]
    if band == "sem":
        half = std / np.sqrt(n)
    elif band == "half":
        half = 0.5 * std
    else:  # "std"
        half = std
    return np.arange(min_len), mean, half


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


def get_scalar_values(data, config_name, key):
    """Valori scalari (non-history) per i seed disponibili."""
    vals = []
    for s in SEEDS:
        d = data[config_name][s]
        if d is not None and key in d and d[key] is not None:
            vals.append(d[key])
    return vals


def moving_average(arr, window=30):
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def _std(vals):
    """std campionaria; 0 se meno di 2 valori."""
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def makedirs(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# GRAFICI TEMPORALI (con banda di varianza selettiva)
# ─────────────────────────────────────────────────────────────────────────────

def plot_temporal(data, group, metric_key, title, ylabel, output_path,
                  window=30, focus_labels=None, thin_labels=None,
                  band_labels=None):
    """Una curva per configurazione (media sui seed, smoothed).

    band_labels: insieme di label su cui disegnare la banda di varianza.
                 Se None (default) la banda viene messa su TUTTE le curve
                 del gruppo (escluse le thin, che non la ricevono mai).
                 Passare una lista di label specifiche per limitarla a quelle.
                 Passare [] (lista vuota) per non disegnare alcuna banda.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    focus_labels = set(focus_labels) if focus_labels else {"Oracle"}
    thin_labels = set(thin_labels) if thin_labels else set()
    # banda: default = tutte le curve del gruppo; mai sulle thin
    if band_labels is None:
        # ricaviamo le label di tutto il gruppo al volo
        band_labels = {CONFIGS[cn]["label"] for cn in group}
    else:
        band_labels = set(band_labels)

    line_ends = []
    max_x = 0

    for config_name in group:
        cfg = CONFIGS[config_name]
        eps, mean, half = get_mean_band_history(data, config_name, metric_key)
        if eps is None:
            print(f"  [SKIP] {config_name} — {metric_key} non disponibile")
            continue

        sm = moving_average(mean, window)
        sm_half = moving_average(half, window)
        x = eps[:len(sm)]
        if len(x) == 0:
            continue

        label = cfg["label"]
        is_focus = label in focus_labels
        is_thin = label in thin_labels

        if is_thin:
            linewidth = 1.15
        elif is_focus:
            linewidth = 2.05
        else:
            linewidth = 1.55
        alpha = 0.85 if is_thin else (1.0 if is_focus else 0.78)
        zorder = 4 if is_focus and not is_thin else 3

        # banda di varianza SOLO su curve designate e mai sulle thin
        if (label in band_labels) and (not is_thin):
            n = min(len(sm), len(sm_half))
            ax.fill_between(
                x[:n], (sm[:n] - sm_half[:n]), (sm[:n] + sm_half[:n]),
                color=cfg["color"], alpha=0.13, linewidth=0, zorder=2,
            )

        ax.plot(
            x, sm,
            color=cfg["color"], linestyle=cfg["ls"],
            linewidth=linewidth, alpha=alpha, label=label, zorder=zorder,
        )

        max_x = max(max_x, int(x[-1]))
        line_ends.append({
            "label": label, "x": float(x[-1]), "y": float(sm[-1]),
            "color": cfg["color"], "alpha": alpha, "is_focus": is_focus,
        })

    ax.set_xlabel("Episodio")
    ax.set_ylabel(ylabel)

    if line_ends:
        ax.set_xlim(0, max_x + 55)
        ymin, ymax = ax.get_ylim()
        min_sep = (ymax - ymin) * 0.035
        ordered = sorted(line_ends, key=lambda item: item["y"])
        adjusted = []
        last_y = None
        for item in ordered:
            y = item["y"] if last_y is None else max(item["y"], last_y + min_sep)
            adjusted.append((item, y))
            last_y = y
        overflow = adjusted[-1][1] - ymax if adjusted else 0
        if overflow > 0:
            adjusted = [(item, y - overflow) for item, y in adjusted]
        for item, y_text in adjusted:
            ax.text(
                max_x + 8, y_text, item["label"],
                color=item["color"], alpha=item["alpha"], fontsize=8.5,
                fontweight="bold" if item["is_focus"] and item["alpha"] >= 0.99 else "normal",
                va="center", clip_on=False,
            )

    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.22),
        ncol=min(len(group), 4), framealpha=0.85,
    )

    fig.tight_layout()
    makedirs(output_path)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Salvato: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 2×2 — confronto per coppie con banda di varianza
# ─────────────────────────────────────────────────────────────────────────────

def plot_2x2(data, panels, metric_key, suptitle, ylabel, output_path,
             window=30, reference_label="Oracle", band_kind=BAND_KIND):
    """Figura 2×2: ogni pannello confronta 2-3 configurazioni + riferimento.

    Parametri
    ----------
    panels          : lista di 4 liste di config_name (una per pannello).
    metric_key      : chiave JSON della history.
    suptitle        : ignorato (titoli rimossi per uso in tesi con didascalia).
    ylabel          : etichetta asse Y.
    reference_label : label della curva di riferimento (no banda, tratteggiata).
    band_kind       : "sem" | "std" | "half".

    Note sulle bande:
    - La curva reference non riceve banda.
    - Le altre curve ricevono una banda fill_between + un hatch leggero
      alternato (prima curva: nessun hatch; seconda curva: hatch "///").
      Questo rende le bande distinguibili anche quando si sovrappongono,
      senza ambiguità di colore.
    - L'asse Y viene mostrato su tutti e 4 i pannelli (non solo quelli
      di sinistra) per rendere immediata la lettura dei valori.
    """
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    # hatch alternato per le curve non-reference: prima piena, seconda tratteggiata
    band_hatches = [None, "///"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    axes_flat = axes.flatten()

    for idx, (ax, group) in enumerate(zip(axes_flat, panels)):
        hatch_counter = 0  # conta solo le curve non-reference

        for config_name in group:
            cfg = CONFIGS[config_name]
            label = cfg["label"]
            is_ref = (label == reference_label)

            histories = [
                data[config_name][s][metric_key]
                for s in SEEDS
                if data[config_name][s] is not None
                and data[config_name][s].get(metric_key)
            ]
            if not histories:
                continue

            min_len = min(len(h) for h in histories)
            arr = np.array([h[:min_len] for h in histories])
            mean = arr.mean(axis=0)
            std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
            n = arr.shape[0]
            if band_kind == "sem":
                half = std / np.sqrt(n)
            elif band_kind == "half":
                half = 0.5 * std
            else:
                half = std

            sm      = moving_average(mean, window)
            sm_half = moving_average(half, window)
            x = np.arange(len(sm))

            lw    = 1.3 if is_ref else 1.8
            alpha = 0.75 if is_ref else 1.0
            ls    = "--" if is_ref else cfg["ls"]

            if not is_ref:
                n2 = min(len(sm), len(sm_half))
                hatch = band_hatches[hatch_counter % len(band_hatches)]
                # fill solido molto leggero come base
                ax.fill_between(
                    x[:n2],
                    sm[:n2] - sm_half[:n2],
                    sm[:n2] + sm_half[:n2],
                    color=cfg["color"], alpha=0.10, linewidth=0, zorder=2,
                )
                # hatch sovrapposto: visibile ma non opaco, distingue le bande
                if hatch:
                    ax.fill_between(
                        x[:n2],
                        sm[:n2] - sm_half[:n2],
                        sm[:n2] + sm_half[:n2],
                        facecolor="none", edgecolor=cfg["color"],
                        hatch=hatch, alpha=0.35, linewidth=0, zorder=3,
                    )
                hatch_counter += 1

            ax.plot(x, sm, color=cfg["color"], linestyle=ls,
                    linewidth=lw, alpha=alpha, label=label, zorder=4)

        ax.set_title(panel_labels[idx], loc="left", fontsize=10,
                     fontweight="500", pad=4)
        ax.set_xlabel("Episodio", fontsize=9)
        # mostra i numeri Y su tutti i pannelli, non solo quelli di sinistra
        ax.tick_params(labelleft=True)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8, loc="lower right", framealpha=0.85)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    makedirs(output_path)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Salvato: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# BAR PLOT — collisioni (barre di errore ±1 std, senza punti seed)
# ─────────────────────────────────────────────────────────────────────────────

def plot_collisions_barplot(data, group, output_path):
    labels, means, stds, colors = [], [], [], []

    for config_name in group:
        cfg = CONFIGS[config_name]
        vals = get_scalar_values(data, config_name, "mean_collisions")
        if not vals:
            continue
        labels.append(cfg["label"])
        means.append(float(np.mean(vals)))
        stds.append(_std(vals))
        colors.append(cfg["color"])

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x, means, color=colors, width=0.55, alpha=0.85, zorder=2)
    ax.errorbar(x, means, yerr=stds, fmt="none", color="black",
                capsize=5, linewidth=1.4, zorder=4)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(x[i], m + s + max(means) * 0.02,
                f"{m:.1f}±{s:.1f}", ha="center", va="bottom",
                fontsize=8.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Collisioni medie per episodio")
    fig.tight_layout()
    makedirs(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvato: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# BAR PLOT — episodi a target (barre di errore ±1 std, senza punti seed)
# ─────────────────────────────────────────────────────────────────────────────

def plot_episodes_to_target(data, group, output_path):
    max_ep = 500
    labels = []
    m075, s075, m080, s080, colors = [], [], [], [], []

    for config_name in group:
        cfg = CONFIGS[config_name]
        ep075 = get_scalar_values(data, config_name, "episodes_to_075")
        ep080 = get_scalar_values(data, config_name, "episodes_to_080")
        if not ep075:
            continue
        labels.append(cfg["label"])
        m075.append(float(np.mean(ep075)))
        s075.append(_std(ep075))
        m080.append(float(np.mean(ep080)) if ep080 else max_ep)
        s080.append(_std(ep080) if ep080 else 0.0)
        colors.append(cfg["color"])

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.bar(x - w/2, m075, width=w, color=colors, alpha=0.9,
           label="Target 0.75", zorder=2)
    ax.errorbar(x - w/2, m075, yerr=s075, fmt="none", color="black",
                capsize=4, linewidth=1.2, zorder=4)
    ax.bar(x + w/2, m080, width=w, color=colors, alpha=0.55,
           label="Target 0.80", zorder=2, hatch="//")
    ax.errorbar(x + w/2, m080, yerr=s080, fmt="none", color="black",
                capsize=4, linewidth=1.2, zorder=4)

    for i, (v75, sv75, v80, sv80) in enumerate(zip(m075, s075, m080, s080)):
        if v75 >= max_ep:
            ax.text(x[i] - w/2, max_ep + 8, "n.r.",
                    ha="center", fontsize=8, color="gray")
        else:
            ax.text(x[i] - w/2, v75 + sv75 + max_ep * 0.02,
                    f"{v75:.0f}±{sv75:.0f}", ha="center", va="bottom",
                    fontsize=7.5, color="#333333")
        if v80 >= max_ep:
            ax.text(x[i] + w/2, max_ep + 8, "n.r.",
                    ha="center", fontsize=8, color="gray")
        else:
            ax.text(x[i] + w/2, v80 + sv80 + max_ep * 0.02,
                    f"{v80:.0f}±{sv80:.0f}", ha="center", va="bottom",
                    fontsize=7.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Episodio (media ± std, 3 seed)")
    ax.set_ylim(0, max_ep * 1.18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, framealpha=0.85)
    fig.tight_layout()
    makedirs(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvato: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SCATTER 8.4 — un punto medio per configurazione, barre di errore su x e y
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_vs_coverage(data, group, output_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    legend_handles = []

    for config_name in group:
        cfg = CONFIGS[config_name]
        covs = [v for v in get_final_from_history(data, config_name, "coverage_history") if v is not None]
        accs = [v for v in get_final_from_history(data, config_name, "accuracy_visited_history") if v is not None]
        if not covs or not accs:
            continue

        mc, sc = float(np.mean(covs)), _std(covs)
        ma, sa = float(np.mean(accs)), _std(accs)

        ax.errorbar(
            mc, ma, xerr=sc, yerr=sa, fmt="o", color=cfg["color"],
            markersize=9, capsize=5, linewidth=1.4,
            markeredgecolor="white", markeredgewidth=0.8, zorder=4,
        )
        ax.annotate(cfg["label"], xy=(mc, ma), xytext=(6, 4),
                    textcoords="offset points", fontsize=8.5,
                    color=cfg["color"], zorder=5)
        legend_handles.append(mpatches.Patch(color=cfg["color"], label=cfg["label"]))

    ax.set_xlabel("Coverage finale (media ± std)")
    ax.set_ylabel("Visited accuracy finale (media ± std)")
    ax.text(0.02, 0.03, "Barre: ±1 std (3 seed)", transform=ax.transAxes,
            fontsize=8, color="#666666", style="italic")
    ax.legend(handles=legend_handles, loc="center left",
              bbox_to_anchor=(1.02, 0.5), fontsize=8.5, framealpha=0.85)
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
        vis = [v for v in get_final_from_history(data, config_name, "accuracy_visited_history") if v is not None]
        unv = [v for v in get_final_from_history(data, config_name, "accuracy_unvisited_history") if v is not None]
        cov = [v for v in get_final_from_history(data, config_name, "coverage_history") if v is not None]
        rows.append({
            "Configurazione":     cfg["label"],
            "Visited accuracy":   f"{np.mean(vis):.3f}" if vis else "n/a",
            "Unvisited accuracy": f"{np.mean(unv):.3f}" if unv else "n/a",
            "Coverage":           f"{np.mean(cov):.3f}" if cov else "n/a",
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
        vis = [v for v in get_final_from_history(data, config_name, "accuracy_visited_history") if v is not None]
        unv = [v for v in get_final_from_history(data, config_name, "accuracy_unvisited_history") if v is not None]
        cov = [v for v in get_final_from_history(data, config_name, "coverage_history") if v is not None]
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
            vis = f"{d['accuracy_visited_history'][-1]:.3f}"   if d and d.get("accuracy_visited_history")   else "—"
            unv = f"{d['accuracy_unvisited_history'][-1]:.3f}" if d and d.get("accuracy_unvisited_history") else "—"
            cov = f"{d['coverage_history'][-1]:.3f}"           if d and d.get("coverage_history")           else "—"
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
# SCATTER APPENDICE — punti per seed (resta col dettaglio seed)
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_vs_coverage_seed(data, group, output_path):
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
            ax.scatter(cov, acc, color=cfg["color"], s=55, alpha=0.8, zorder=3)
            ax.annotate(str(seed), xy=(cov, acc), xytext=(4, 4),
                        textcoords="offset points", fontsize=7.5,
                        color=cfg["color"], zorder=4)
        if has_points:
            legend_handles.append(mpatches.Patch(color=cfg["color"], label=cfg["label"]))

    ax.set_xlabel("Coverage finale")
    ax.set_ylabel("Visited accuracy finale")
    ax.legend(handles=legend_handles, loc="center left",
              bbox_to_anchor=(1.02, 0.5), fontsize=8.5, framealpha=0.85)
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
    parser.add_argument("--results_dir", default="results3")
    parser.add_argument("--output_dir",  default="thesis_plots2")
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
    print("\n── 8.2 Ablation study ──────────────────────────────────")
    d82 = os.path.join(O, "8.2_ablation")
    # Global accuracy: curve che si incrociano -> banda su tutte (default)
    plot_temporal(data, GROUP_82, "accuracy_history",
                  "Global accuracy — Ablation study",
                  "Global accuracy (media 3 seed)",
                  os.path.join(d82, "global_accuracy.png"),
                  focus_labels=["Bayesian", "Bayesian + GF"],
                  thin_labels=["No Belief", "Oracle"])
    # Unvisited: banda su tutte (default)
    plot_temporal(data, GROUP_82, "accuracy_unvisited_history",
                  "Unvisited accuracy — Ablation study",
                  "Unvisited accuracy (media 3 seed)",
                  os.path.join(d82, "unvisited_accuracy.png"),
                  focus_labels=["Bayesian", "Bayesian + GF"],
                  thin_labels=["No Belief", "Oracle"])
    # Coverage: banda su tutte (default)
    plot_temporal(data, GROUP_82, "coverage_history",
                  "Coverage — Ablation study",
                  "Coverage (media 3 seed)",
                  os.path.join(d82, "coverage.png"),
                  window=45,
                  focus_labels=["Bayesian", "Bayesian + GF"],
                  thin_labels=["No Belief", "Oracle"])

    # ── 8.3 Analisi LSTM ────────────────────────────────────────────────────
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
    print("\n── 8.4 Trade-off e comportamento ───────────────────────")
    d84 = os.path.join(O, "8.4_tradeoff_behavior")
    plot_collisions_barplot(data, GROUP_84,
                            os.path.join(d84, "collisions_barplot.png"))
    plot_episodes_to_target(data, GROUP_84,
                            os.path.join(d84, "episodes_to_target.png"))
    plot_accuracy_vs_coverage(data, GROUP_84,
                              os.path.join(d84, "accuracy_vs_coverage.png"))

    # ── 8.5 Pannelli 2x2 ────────────────────────────────────────────────────
    print("\n── 8.5 Pannelli 2x2 ────────────────────────────────────")
    d85 = os.path.join(O, "8.5_2x2_panels")

    # Layout narrativo (ablation study progressivo):
    #   (a) baseline: cosa succede senza GF
    #   (b) effetto GF applicato al Bayesian
    #   (c) LSTM puro vs miglior baseline con GF
    #   (d) metodo proposto (LSTM+GF) vs miglior baseline con GF
    panels_2x2 = [
        ["2_mappo_no_belief", "3_mappo_bayesian",  "7_oracle_conf"],  # (a)
        ["3_mappo_bayesian",  "4_mappo_bayes_gf",  "7_oracle_conf"],  # (b)
        ["4_mappo_bayes_gf",  "5_mappo_lstm",      "7_oracle_conf"],  # (c)
        ["4_mappo_bayes_gf",  "6_mappo_gf_lstm",   "7_oracle_conf"],  # (d)
    ]

    plot_2x2(data, panels_2x2,
             metric_key="accuracy_unvisited_history",
             suptitle="Unvisited accuracy — confronto per coppie (±SEM, 3 seed)",
             ylabel="Unvisited accuracy",
             output_path=os.path.join(d85, "unvisited_2x2.png"),
             band_kind="sem")

    plot_2x2(data, panels_2x2,
             metric_key="coverage_history",
             suptitle="Coverage — confronto per coppie (±SEM, 3 seed)",
             ylabel="Coverage",
             output_path=os.path.join(d85, "coverage_2x2.png"),
             window=45,
             band_kind="sem")

    plot_2x2(data, panels_2x2,
             metric_key="accuracy_history",
             suptitle="Global accuracy — confronto per coppie (±SEM, 3 seed)",
             ylabel="Global accuracy",
             output_path=os.path.join(d85, "global_accuracy_2x2.png"),
             band_kind="sem")

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