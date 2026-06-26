from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUTPUT_DIR = SCRIPT_DIR / "img3"

C_GRID       = "#F0F4F8"
C_VISITED_L  = "#B8D4E8"
C_VISITED    = "#2E75B6"
C_AGENT      = "#C00000"
C_BORDER     = "#CCCCCC"
C_TEXT       = "#1A1A2E"
C_ALIGN_HI   = "#2E75B6"
C_ALIGN_LO   = "#E07B39"
C_PLANT      = "#4A7C59"
C_DRONE      = "#C00000"
C_BAR_HI     = "#2E75B6"
C_BAR_LO     = "#E07B39"
C_PROPAGATED = "#7EB8D9"
K   = 10
DPI = 200

# 10 colori distinti per le classi 0-9
CLASS_COLORS = [
    "#264653", "#2A9D8F", "#8AB17D", "#E9C46A", "#F4A261",
    "#E76F51", "#A8DADC", "#457B9D", "#6A4C93", "#C77DFF",
]

def save_figure(fig, filename, output_dir=OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"{filename}.{ext}", bbox_inches="tight", dpi=DPI)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3.1 — Struttura della griglia: classi e orientamenti
# ─────────────────────────────────────────────────────────────────────────────
def create_fig31(output_dir=OUTPUT_DIR):
    """
    Mostra la struttura del campo:
    - sinistra: mappa delle classi (colori 0-9) su griglia 20x20
    - destra: frecce degli orientamenti delle piante sulla stessa griglia
    """
    np.random.seed(42)

    # genera un campo sintetico con correlazione spaziale
    # usiamo un approccio semplice: blocchi contigui per simulare len_scale=5
    SIZE = 20
    grid_classes = np.zeros((SIZE, SIZE), dtype=int)
    grid_angles  = np.zeros((SIZE, SIZE))

    # genera regioni contigue (simula correlazione spaziale)
    # partiziona la griglia in blocchi ~5x5 con classe e angolo omogenei
    block = 5
    for br in range(0, SIZE, block):
        for bc in range(0, SIZE, block):
            cls = np.random.randint(0, K)
            ang = np.random.uniform(-np.pi, np.pi)
            for r in range(br, min(br + block, SIZE)):
                for c in range(bc, min(bc + block, SIZE)):
                    # piccola variazione locale
                    grid_classes[r, c] = np.clip(
                        cls + np.random.randint(-1, 2), 0, K-1)
                    grid_angles[r, c]  = ang + np.random.uniform(-0.3, 0.3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), facecolor="white")

    # ── SINISTRA: mappa delle classi ────────────────────────────────────────
    cmap = ListedColormap(CLASS_COLORS)
    img = ax1.imshow(grid_classes, cmap=cmap, vmin=0, vmax=K-1,
                     origin="upper", interpolation="nearest")

    # griglia leggera
    for i in range(SIZE + 1):
        ax1.axhline(i - 0.5, color="white", linewidth=0.4)
        ax1.axvline(i - 0.5, color="white", linewidth=0.4)

    ax1.set_title("Classi delle celle (ground truth)", fontsize=11,
                  fontweight="bold", color=C_TEXT, pad=8)
    ax1.set_xlabel("Colonna", fontsize=10, color=C_TEXT)
    ax1.set_ylabel("Riga", fontsize=10, color=C_TEXT)
    ax1.tick_params(labelsize=8)

    # colorbar discreta
    cbar = fig.colorbar(img, ax=ax1, ticks=range(K),
                        fraction=0.046, pad=0.04)
    cbar.set_label("Classe (0–9)", fontsize=9, color=C_TEXT)
    cbar.ax.tick_params(labelsize=8)

    # nota "nascosta agli agenti"
    ax1.text(0.5, -0.12,
             "⚠ Il ground truth è nascosto agli agenti",
             transform=ax1.transAxes, ha="center", fontsize=8.5,
             color="#888888", style="italic")

    # ── DESTRA: orientamenti delle piante ───────────────────────────────────
    # sfondo con classi in trasparenza
    ax2.imshow(grid_classes, cmap=cmap, vmin=0, vmax=K-1,
               origin="upper", interpolation="nearest", alpha=0.25)

    # frecce orientamento — una per cella
    step = 1
    for r in range(0, SIZE, step):
        for c in range(0, SIZE, step):
            ang = grid_angles[r, c]
            dx = np.cos(ang) * 0.38
            dy = np.sin(ang) * 0.38
            # matplotlib: asse y invertito con imshow, quindi invertiamo dy
            ax2.annotate("",
                xy=(c + dx, r - dy),
                xytext=(c - dx, r + dy),
                arrowprops=dict(
                    arrowstyle="<->",
                    color=C_PLANT,
                    lw=0.9,
                    mutation_scale=6,
                ),
                zorder=3,
            )

    for i in range(SIZE + 1):
        ax2.axhline(i - 0.5, color="white", linewidth=0.4)
        ax2.axvline(i - 0.5, color="white", linewidth=0.4)

    ax2.set_xlim(-0.5, SIZE - 0.5)
    ax2.set_ylim(SIZE - 0.5, -0.5)
    ax2.set_title("Orientamento dei filari per cella", fontsize=11,
                  fontweight="bold", color=C_TEXT, pad=8)
    ax2.set_xlabel("Colonna", fontsize=10, color=C_TEXT)
    ax2.set_ylabel("Riga", fontsize=10, color=C_TEXT)
    ax2.tick_params(labelsize=8)

    ax2.text(0.5, -0.12,
             "Le celle vicine hanno orientamenti simili (correlazione spaziale)",
             transform=ax2.transAxes, ha="center", fontsize=8.5,
             color="#888888", style="italic")

    fig.suptitle("Fig. 3.1 — Struttura del campo: classi e orientamenti delle piante",
                 fontsize=12, fontweight="bold", color=C_TEXT, y=1.01)
    plt.tight_layout()
    save_figure(fig, "fig31_field_structure", output_dir)
    print(f"Fig 3.1 salvata in: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3.2 — Meccanismo di osservazione locale (zoom patch + sensore)
# ─────────────────────────────────────────────────────────────────────────────
def create_fig32_observation(output_dir=OUTPUT_DIR):
    """
    Spiega il meccanismo di osservazione:
    - sinistra: zoom patch 3x3 con agente, direzione, alignment per cella
    - destra: distribuzione del sensore per due celle (alta/bassa alignment)
    """
    fig = plt.figure(figsize=(11, 5), facecolor="white")
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.2], wspace=0.45)

    # ── SINISTRA: zoom patch 3x3 ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(-0.5, 2.5)
    ax1.set_facecolor("white")
    ax1.set_aspect("equal")

    # valori di alignment per le 9 celle
    # agente al centro (1,1), si muove verso EST (direzione 0°)
    alignments = np.array([
        [0.12, 0.08, 0.15],   # row 0: celle in alto
        [0.88, 1.00, 0.91],   # row 1: celle centrali (filare orizzontale)
        [0.10, 0.07, 0.13],   # row 2: celle in basso
    ])
    # classe vera per ogni cella (inventata per illustrazione)
    true_classes = np.array([
        [3, 3, 4],
        [7, 7, 7],
        [3, 4, 3],
    ])

    for r in range(3):
        for c in range(3):
            al = alignments[r, c]
            color = C_ALIGN_HI if al >= 0.5 else C_ALIGN_LO
            alpha = 0.20 + 0.70 * al
            ax1.add_patch(patches.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                facecolor=color, alpha=alpha,
                edgecolor="white", linewidth=2.0, zorder=1))

            # alignment value
            ax1.text(c, r + 0.28, f"align={al:.2f}",
                     ha="center", va="center", fontsize=8, fontweight="bold",
                     color="white" if al > 0.4 else C_TEXT, zorder=3)
            # classe vera (piccola, sotto)
            ax1.text(c, r - 0.22, f"classe={true_classes[r,c]}",
                     ha="center", va="center", fontsize=7,
                     color="white" if al > 0.4 else "#555555", zorder=3)

    # agente al centro con freccia direzione EST
    ax1.plot(1, 1, "o", color=C_AGENT, markersize=16,
             markeredgecolor="white", markeredgewidth=2.0, zorder=5)
    ax1.annotate("", xy=(1.85, 1), xytext=(1.35, 1),
                 arrowprops=dict(arrowstyle="-|>", color=C_AGENT,
                                 lw=2.2, mutation_scale=18), zorder=6)
    ax1.text(2.3, 1, "EST", ha="left", va="center",
             fontsize=8, color=C_AGENT, fontweight="bold")

    # etichette assi
    ax1.set_xticks([0, 1, 2])
    ax1.set_yticks([0, 1, 2])
    ax1.set_xticklabels(["col−1", "col", "col+1"], fontsize=8)
    ax1.set_yticklabels(["row−1", "row", "row+1"], fontsize=8)
    ax1.set_title("Patch 3×3 osservata\n(agente si muove verso EST)",
                  fontsize=10, color=C_TEXT, fontweight="bold", pad=8)

    # nota: filare orizzontale → alta alignment sulla riga centrale
    ax1.text(1, -0.42,
             "Il filare è orizzontale → alignment alto sulla riga centrale",
             ha="center", va="center", fontsize=8, color="#555555",
             style="italic")

    # ── DESTRA: confronto distribuzioni sensore ──────────────────────────────
    ax_hi = fig.add_subplot(gs[1])
    gs_inner = GridSpec(2, 1, figure=fig,
                        left=0.60, right=0.97,
                        top=0.88, bottom=0.12,
                        hspace=0.60)
    ax_hi  = fig.add_subplot(gs_inner[0])
    ax_lo  = fig.add_subplot(gs_inner[1])

    def sensor_dist(alignment, true_class, K=10):
        noise = 1.0 - alignment
        p = np.ones(K) * noise / (K - 1)
        p[true_class] = alignment + noise / K
        return p / p.sum()

    classes = np.arange(K)

    # cella centrale (1,1): alignment=1.00, classe=7
    p_hi = sensor_dist(1.00, 7)
    ax_hi.bar(classes, p_hi,
              color=[C_BAR_HI if i == 7 else "#AACCE8" for i in classes],
              edgecolor="white", linewidth=0.6)
    ax_hi.set_title("Cella centrale — align=1.00, classe vera=7",
                    fontsize=8.5, color=C_TEXT, fontweight="bold", pad=4)
    ax_hi.set_xticks(classes)
    ax_hi.set_xticklabels([str(i) for i in classes], fontsize=7)
    ax_hi.set_ylim(0, 1.05)
    ax_hi.set_yticks([0, 0.5, 1.0])
    ax_hi.set_yticklabels(["0", ".5", "1"], fontsize=7)
    ax_hi.spines["top"].set_visible(False)
    ax_hi.spines["right"].set_visible(False)
    ax_hi.set_facecolor("white")
    ax_hi.text(7, 0.75, "↑ concentrata\n  sulla classe vera",
               fontsize=7.5, color=C_BAR_HI)

    # cella angolo (0,0): alignment=0.12, classe=3
    p_lo = sensor_dist(0.12, 3)
    ax_lo.bar(classes, p_lo,
              color=[C_BAR_LO if i == 3 else "#F5C9A8" for i in classes],
              edgecolor="white", linewidth=0.6)
    ax_lo.set_title("Cella angolo — align=0.12, classe vera=3",
                    fontsize=8.5, color=C_TEXT, fontweight="bold", pad=4)
    ax_lo.set_xticks(classes)
    ax_lo.set_xticklabels([str(i) for i in classes], fontsize=7)
    ax_lo.set_ylim(0, 1.05)
    ax_lo.set_yticks([0, 0.5, 1.0])
    ax_lo.set_yticklabels(["0", ".5", "1"], fontsize=7)
    ax_lo.spines["top"].set_visible(False)
    ax_lo.spines["right"].set_visible(False)
    ax_lo.set_facecolor("white")
    ax_lo.set_xlabel("Classe", fontsize=8)
    ax_lo.text(0.5, 0.72, "quasi uniforme\n(molto rumorosa)",
               fontsize=7.5, color=C_BAR_LO)

    fig.suptitle("Fig. 3.2 — Meccanismo di osservazione locale: alignment e distribuzione sensore",
                 fontsize=11, fontweight="bold", color=C_TEXT, y=1.01)

    save_figure(fig, "fig32_observation", output_dir)
    print(f"Fig 3.2 salvata in: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3.3 — Occlusione direzionale (tre pannelli)
# ─────────────────────────────────────────────────────────────────────────────
def draw_cell_panel(ax, angle_deg, drone_dir_deg, alignment, label, color_bar):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_facecolor("#F7F9FC")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#CCCCCC"); sp.set_linewidth(0.8)
    cx, cy = 0.5, 0.58
    ang_rad = np.radians(angle_deg)
    L = 0.32
    dx_p, dy_p = np.cos(ang_rad)*L, np.sin(ang_rad)*L
    ax.annotate("", xy=(cx+dx_p, cy+dy_p), xytext=(cx-dx_p, cy-dy_p),
                arrowprops=dict(arrowstyle="<->", color="#4A7C59", lw=2.5))
    ax.text(cx+dx_p+0.05, cy+dy_p, "filare",
            fontsize=7.5, color="#4A7C59", ha="left", va="center")
    drone_rad = np.radians(drone_dir_deg)
    Ld = 0.26
    dx_d, dy_d = np.cos(drone_rad)*Ld, np.sin(drone_rad)*Ld
    ax.annotate("", xy=(cx+dx_d, cy+dy_d), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color=C_DRONE,
                                lw=2.2, mutation_scale=16))
    ax.plot(cx, cy, "o", color=C_DRONE, markersize=9,
            markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    ax.text(cx+dx_d+0.04, cy+dy_d, "drone",
            fontsize=7.5, color=C_DRONE, ha="left", va="center")
    ax.text(0.5, 0.12, f"alignment = {alignment:.2f}",
            fontsize=9.5, ha="center", va="center", color="white",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor=color_bar, alpha=0.90, edgecolor="none"))
    ax.set_title(label, fontsize=9.5, color=C_TEXT, fontweight="bold", pad=5)


def draw_sensor_panel(ax, alignment, true_class=4):
    classes = np.arange(K)
    noise = 1.0 - alignment
    probs = np.ones(K) * noise / (K-1)
    probs[true_class] = alignment + noise/K
    probs /= probs.sum()
    colors = ["#2E75B6" if i==true_class else "#AACCE8" for i in classes]
    ax.bar(classes, probs, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_xlim(-0.5, K-0.5); ax.set_ylim(0, 1.05)
    ax.set_xticks(classes)
    ax.set_xticklabels([str(i) for i in classes], fontsize=7)
    ax.set_yticks([0,0.5,1.0]); ax.set_yticklabels(["0",".5","1"], fontsize=7)
    ax.set_xlabel("Classe", fontsize=8)
    ax.set_title("Distribuzione sensore", fontsize=8.5, pad=3, color=C_TEXT)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")


def create_fig33_occlusion(output_dir=OUTPUT_DIR):
    scenarios = [
        dict(angle=30, drone=30,  alignment=0.97,
             label="Drone allineato al filare",      color="#2E75B6"),
        dict(angle=30, drone=75,  alignment=0.50,
             label="Drone parzialmente allineato",   color="#D4650A"),
        dict(angle=30, drone=120, alignment=0.10,
             label="Drone perpendicolare al filare", color="#E07B39"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(9, 7.5),
        gridspec_kw={"width_ratios":[1,1.1], "wspace":0.35, "hspace":0.55})
    for i, sc in enumerate(scenarios):
        draw_cell_panel(axes[i][0], sc["angle"], sc["drone"],
                        sc["alignment"], sc["label"], sc["color"])
        draw_sensor_panel(axes[i][1], sc["alignment"])
    fig.suptitle("Fig. 3.3 — Occlusione direzionale: effetto dell'alignment sul sensore",
                 fontsize=12, fontweight="bold", color=C_TEXT, y=1.01)
    save_figure(fig, "fig33_occlusion", output_dir)
    print(f"Fig 3.3 salvata in: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3.4 — Tre tipi di cella (semplificata)
# ─────────────────────────────────────────────────────────────────────────────
def create_fig34(output_dir=OUTPUT_DIR):
    GRID = 12
    observed = set()
    for r in range(4, 8):
        for c in range(2, 10):
            observed.add((r, c))
    for r in range(1, 4):
        for c in range(8, 11):
            observed.add((r, c))

    propagated = set()
    for (r, c) in observed:
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = r+dr, c+dc
                if 0 <= nr < GRID and 0 <= nc < GRID and (nr,nc) not in observed:
                    propagated.add((nr, nc))

    prior = set()
    for r in range(GRID):
        for c in range(GRID):
            if (r,c) not in observed and (r,c) not in propagated:
                prior.add((r,c))

    fig, ax = plt.subplots(figsize=(6.5, 6.5), facecolor="white")
    ax.set_xlim(-0.5, GRID-0.5); ax.set_ylim(-0.5, GRID-0.5)
    ax.set_facecolor(C_GRID)

    for (r,c) in prior:
        ax.add_patch(patches.Rectangle((c-.5,r-.5),1,1,
            facecolor=C_GRID, edgecolor=C_BORDER, linewidth=0.4, zorder=1))
    for (r,c) in propagated:
        ax.add_patch(patches.Rectangle((c-.5,r-.5),1,1,
            facecolor=C_PROPAGATED, edgecolor="white", linewidth=0.4,
            zorder=2, alpha=0.80))
    for (r,c) in observed:
        ax.add_patch(patches.Rectangle((c-.5,r-.5),1,1,
            facecolor=C_VISITED, edgecolor="white", linewidth=0.4, zorder=3))

    for i in range(GRID+1):
        ax.axhline(i-.5, color=C_BORDER, linewidth=0.4, zorder=0)
        ax.axvline(i-.5, color=C_BORDER, linewidth=0.4, zorder=0)

    ax.text(5.5, 5.5, "Osservata\ndirettamente",
            ha="center", va="center", fontsize=10, fontweight="bold",
            color="white", zorder=5)
    ax.text(9.5, 2.5, "Osservata",
            ha="center", va="center", fontsize=8, fontweight="bold",
            color="white", zorder=5)
    ax.text(1.2, 5.5, "Propagata\nvia GF",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=C_TEXT, zorder=5,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor=C_PROPAGATED, alpha=0.85, edgecolor="none"))
    ax.text(1.5, 10.5, "Prior\nuniforme",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="#666666", zorder=5,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor=C_GRID, alpha=0.9, edgecolor=C_BORDER))

    ax.set_xticks(range(GRID)); ax.set_yticks(range(GRID))
    ax.tick_params(labelsize=7, length=2)
    ax.set_xlabel("Colonna", fontsize=10, color=C_TEXT)
    ax.set_ylabel("Riga", fontsize=10, color=C_TEXT)

    n_obs=len(observed); n_prop=len(propagated); n_pri=len(prior)
    legend_els = [
        patches.Patch(facecolor=C_VISITED,
                      label=f"Osservata ({n_obs} celle) → Visited Accuracy"),
        patches.Patch(facecolor=C_PROPAGATED,
                      label=f"Propagata GF ({n_prop} celle)", alpha=0.80),
        patches.Patch(facecolor=C_GRID,
                      label=f"Prior uniforme ({n_pri} celle) → Inferred Accuracy",
                      edgecolor=C_BORDER, linewidth=0.8),
    ]
    ax.legend(handles=legend_els, loc="lower right", fontsize=8.5,
              framealpha=0.95, edgecolor="#AAAAAA")

    ax.set_title("Fig. 3.4 — Tre tipi di cella nel protocollo di valutazione",
                 fontsize=11, fontweight="bold", color=C_TEXT, pad=10)
    plt.tight_layout()
    save_figure(fig, "fig34_visited_unvisited", output_dir)
    print(f"Fig 3.4 salvata in: {output_dir}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    create_fig31(OUTPUT_DIR)
    create_fig32_observation(OUTPUT_DIR)
    create_fig33_occlusion(OUTPUT_DIR)
    create_fig34(OUTPUT_DIR)
    print(f"\nCompletato. Figure in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()