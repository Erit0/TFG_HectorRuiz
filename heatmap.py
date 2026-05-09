"""
Heatmap OLD (eje X) vs NEW (eje Y) - Comparación fila a fila
- Se compara estrictamente cada fila de OLD con la misma fila de NEW.
- Métrica: similitud = 1 - mean(|NEW - OLD| / (max(|OLD|,|NEW|) + ε))
- Escala: blanco = muy diferente, azul oscuro = muy parecido (1.0)
- Las features que deberían cambiar según LIMITES_FISICOS se marcan en rojo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys

# ── Features ──────────────────────────────────────────────────────────────────
FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count",
    "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
]

# Features que el GAN debería modificar según LIMITES_FISICOS
GAN_MODIFIED = {
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'Bwd Packet Length Min', 'Fwd Header Length',
    'Bwd Packet Length Std', 'Bwd Packet Length Mean',
    'Avg Bwd Segment Size', 'Bwd Packet Length Max',
    'Packet Length Mean', 'Max Packet Length',
    'Total Length of Bwd Packets', 'Flow IAT Mean',
}

# ── Carga ──────────────────────────────────────────────────────────────────────
csv_path = sys.argv[1] if len(sys.argv) > 1 else "REPORTE_GAN_COMPLETO.csv"
output   = sys.argv[2] if len(sys.argv) > 2 else "heatmap_similitud.pdf"

print(f"Cargando {csv_path}…")
df = pd.read_csv(csv_path)

valid_features = [f for f in FEATURES
                  if f"OLD_{f}" in df.columns and f"NEW_{f}" in df.columns]
print(f"  {len(valid_features)} features válidas · {len(df)} muestras totales en el CSV")

# ── Matriz de similitud (n_features × n_features) ─────────────────────────────
# Celda (i, j): similitud estricta fila a fila entre NEW_i y OLD_j
n = len(valid_features)
sim_matrix = np.full((n, n), np.nan)

for i, fi in enumerate(valid_features):   # eje Y → NEW_fi
    # Extraemos la columna NEW cruda
    nv = pd.to_numeric(df[f"NEW_{fi}"], errors="coerce").values
    
    for j, fj in enumerate(valid_features):  # eje X → OLD_fj
        # Extraemos la columna OLD cruda
        o = pd.to_numeric(df[f"OLD_{fj}"], errors="coerce").values
        
        # ⚠️ CRUCIAL: Máscara para mantener las filas emparejadas y descartar NaNs
        mask = ~np.isnan(nv) & ~np.isnan(o)
        a = nv[mask]
        b = o[mask]
        
        if len(a) < 2:
            continue
            
        denom = np.maximum(np.abs(a), np.abs(b)) + 1e-9
        ratio = np.abs(a - b) / denom
        
        # SIN CAP: Se toma la media real de la variación total (incluyendo errores gigantes)
        ratio = np.clip(ratio, 0, 1.0) # El ratio de variación de esta métrica por definición va de 0 a 1
        
        sim_matrix[i, j] = 1.0 - ratio.mean()

print("\nRango de similitud en la matriz: "
      f"min={np.nanmin(sim_matrix):.4f}  max={np.nanmax(sim_matrix):.4f}")

# ── Plot ───────────────────────────────────────────────────────────────────────
# Tamaño de celda y márgenes ajustados para mejor legibilidad
cell     = 0.5
margin_l = 4.0
margin_b = 4.0
fig_w    = margin_l + n * cell
fig_h    = margin_b + n * cell

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.patch.set_facecolor("#ffffff")
ax.set_facecolor("#ffffff")

im = ax.imshow(
    sim_matrix,
    aspect="auto",
    cmap="Blues",        # blanco=0 (diferente), azul oscuro=1 (parecido)
    vmin=0, vmax=1,      # Forzamos la escala estricta de 0 a 1
    interpolation="nearest",
    origin="upper",
)

# ── Etiquetas ─────────────────────────────────────────────────────────────────
tick_colors = ["#b71c1c" if f in GAN_MODIFIED else "#1a2a3a" for f in valid_features]

ax.set_xticks(range(n))
ax.set_xticklabels(
    [f[:20] + "…" if len(f) > 20 else f for f in valid_features],
    rotation=90, fontsize=8, ha="right", # Fuente ligeramente aumentada a 8
)
for tick, col in zip(ax.get_xticklabels(), tick_colors):
    tick.set_color(col)
    if col == "#b71c1c":
        tick.set_fontweight("bold")

ax.set_yticks(range(n))
ax.set_yticklabels(
    [f[:22] + "…" if len(f) > 22 else f for f in valid_features],
    fontsize=8,
)
for tick, col in zip(ax.get_yticklabels(), tick_colors):
    tick.set_color(col)
    if col == "#b71c1c":
        tick.set_fontweight("bold")

ax.set_xlabel("Columnas  OLD", fontsize=12, color="#0d47a1", labelpad=15, fontweight="bold")
ax.set_ylabel("Columnas  NEW", fontsize=12, color="#0d47a1", labelpad=15, fontweight="bold")
ax.tick_params(axis="both", which="both", length=0, pad=5)

for sp in ax.spines.values():
    sp.set_visible(False)

# Cuadrícula sutil
for x in range(n):
    ax.axvline(x - 0.5, color="#e0e8f0", linewidth=0.5)
for y in range(n):
    ax.axhline(y - 0.5, color="#e0e8f0", linewidth=0.5)

# Diagonal: borde rojo en GAN_MODIFIED, negro fino en el resto
for i, f in enumerate(valid_features):
    color = "#c62828" if f in GAN_MODIFIED else "#37474f"
    lw    = 2.0       if f in GAN_MODIFIED else 0.8
    ax.add_patch(mpatches.Rectangle(
        (i - 0.5, i - 0.5), 1, 1,
        fill=False, edgecolor=color, linewidth=lw, zorder=3
    ))

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.015, shrink=0.75)
cbar.set_label("Similitud fila",
               fontsize=10, color="#1a2a3a", labelpad=10)
cbar.ax.tick_params(labelsize=9, colors="#1a2a3a")
cbar.outline.set_edgecolor("#90aac8")

# Leyenda
legend_elements = [
    mpatches.Patch(facecolor="none", edgecolor="#c62828", linewidth=2.0,
                   label="Modificada por GAN (LIMITES_FISICOS)"),
    mpatches.Patch(facecolor="none", edgecolor="#37474f", linewidth=0.8,
                   label="Sin restricción explícita"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=9,
          framealpha=0.9, edgecolor="#aabbcc", bbox_to_anchor=(1.0, 1.01))

ax.set_title(
    "Mapa de calor de similitud entre columnas OLD vs NEW\n"
    ,fontsize=14, pad=20, fontweight="bold"
)

plt.tight_layout()
plt.savefig(output, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\n[OK] Heatmap guardado en: {output}")

# ── Diagnóstico diagonal ───────────────────────────────────────────────────────
print("\nDiagonal — Similitud estricta OLD_X vs NEW_X (lo que más importa):")
print(f"  {'Feature':<38} {'Similitud':>10}  {'GAN':>6}")
print("  " + "-"*60)
for i, f in enumerate(valid_features):
    v = sim_matrix[i, i]
    marca = "◀ SÍ" if f in GAN_MODIFIED else ""
    flag  = "⚠" if f in GAN_MODIFIED and v > 0.85 else "" # Alerta si una var del GAN no cambió casi nada
    print(f"  {f:<38} {v:>10.4f}  {marca:>6} {flag}")