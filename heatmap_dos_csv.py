"""
Heatmap de similitud entre dos CSVs distintos:
  - OLD = ataques_adversarios_COMPLETOS.csv  (features sin prefijo)
  - NEW = REPORTE_GLOBAL_PREDICCIONES.csv    (features con nombres distintos → se mapean)

Join exacto por posición global de generación:
  - El generador asigna t0 = base + i*200 a cada fila del CSV OLD filtrado.
  - CICFlowMeter hereda ese t0 en el nombre del PCAP (_ISCX.csv).
  - Ordenando NEW por timestamp extraído de Archivo_Origen se recupera
    el orden original → fila i de NEW ordenado == fila i de OLD filtrado.

PCAPs con 2 subflujos (CICFlowMeter los partió):
  - Se reconstruye el flujo completo agregando ambos subflujos.
  - Conteos (pkts, bytes, flags, headers): suma directa.
  - Duraciones e IAT totales: máximo (el flujo dura lo que dura el más largo).
  - Extremos (max/min de longitudes e IATs): max/min entre subflujos.
  - Features derivadas (means, ratios, rates): recalculadas desde los conteos reconstruidos.
  - Stds y variances: aproximadas como media ponderada por paquetes de los subflujos
    (no hay acceso a los valores raw, pero es mejor que ignorar un subflujo entero).
  - Init_Win_bytes: del subflujo cuyo src es el atacante (192.168.1.50) para fwd,
    del subflujo con bwd>0 (o src=victima) para bwd.

Uso:
  python heatmap_dos_csv.py [csv_old] [csv_new] [salida.pdf]
"""

import re
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Mapeo OLD_nombre → NEW_nombre ─────────────────────────────────────────────
COL_MAP = {
    "Total Fwd Packets":            "Total Fwd Packet",
    "Total Backward Packets":       "Total Bwd packets",
    "Total Length of Fwd Packets":  "Total Length of Fwd Packet",
    "Total Length of Bwd Packets":  "Total Length of Bwd Packet",
    "Min Packet Length":            "Packet Length Min",
    "Max Packet Length":            "Packet Length Max",
    "Avg Fwd Segment Size":         "Fwd Segment Size Avg",
    "Avg Bwd Segment Size":         "Bwd Segment Size Avg",
    "Init_Win_bytes_forward":       "FWD Init Win Bytes",
    "Init_Win_bytes_backward":      "Bwd Init Win Bytes",
    "act_data_pkt_fwd":             "Fwd Act Data Pkts",
}

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

GAN_MODIFIED = {
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'Bwd Packet Length Min', 'Fwd Header Length',
    'Bwd Packet Length Std', 'Bwd Packet Length Mean',
    'Avg Bwd Segment Size', 'Bwd Packet Length Max',
    'Packet Length Mean', 'Max Packet Length',
    'Total Length of Bwd Packets', 'Flow IAT Mean',
}

IP_ATACANTE = "192.168.1.50"


# ── Reconstrucción de flujo partido en dos subflujos ──────────────────────────
def _std_ponderada(mean1, std1, n1, mean2, std2, n2):
    """
    Std combinada de dos grupos sin acceso a los valores raw.
    Fórmula exacta para combinar medias y varianzas de dos muestras.
    """
    n   = n1 + n2
    if n <= 1:
        return 0.0
    # varianza combinada: var = [n1*(std1²+mean1²) + n2*(std2²+mean2²)] / n - mean_comb²
    mean_c = (n1 * mean1 + n2 * mean2) / n
    var_c  = (n1 * (std1**2 + mean1**2) + n2 * (std2**2 + mean2**2)) / n - mean_c**2
    return float(np.sqrt(max(var_c, 0.0)))


def reconstruir_flujo(grupo: pd.DataFrame) -> pd.Series:
    """
    Agrega dos subflujos de CICFlowMeter en un único flujo reconstruido.

    Identifica cuál subflujo es el 'principal' (src=atacante, tiene paquetes Bwd)
    y cuál es el 'secundario' (src=victima o solo Fwd), y combina sus features
    según la naturaleza de cada columna.
    """
    # ── Identificar subflujo principal (src=atacante) y secundario ───────────
    mask_atk = grupo["Src IP"] == IP_ATACANTE
    if mask_atk.sum() == 1:
        A = grupo[mask_atk].iloc[0]       # subflujo desde el atacante
        B = grupo[~mask_atk].iloc[0]      # subflujo desde la victima
    else:
        # Ambos tienen el mismo Src IP (caso raro): usar el de más paquetes como A
        pkts = (pd.to_numeric(grupo["Total Fwd Packet"],   errors="coerce").fillna(0)
              + pd.to_numeric(grupo["Total Bwd packets"],  errors="coerce").fillna(0))
        idx_max = pkts.idxmax()
        A = grupo.loc[idx_max]
        B = grupo.drop(index=idx_max).iloc[0]

    def g(row, col, default=0.0):
        return pd.to_numeric(row.get(col, default), errors="coerce") or default

    # ── Conteos de paquetes ───────────────────────────────────────────────────
    # El subflujo A ve correctamente sus Fwd (atacante→victima) y los Bwd (respuesta).
    # El subflujo B ve como "Fwd" los paquetes que la victima mandó de vuelta
    # usando el mismo puerto fuente (los que CICFlowMeter separó en un 2º flujo).
    # En el flujo original OLD, esos paquetes son Bwd adicionales.
    fwd_A   = int(g(A, "Total Fwd Packet"))
    bwd_A   = int(g(A, "Total Bwd packets"))
    fwd_B   = int(g(B, "Total Fwd Packet"))   # son realmente Bwd del flujo original
    bwd_B   = int(g(B, "Total Bwd packets"))  # normalmente 0

    fwd_total = fwd_A                         # solo los del atacante son Fwd reales
    bwd_total = bwd_A + fwd_B + bwd_B         # respuestas de la victima de ambos subflujos

    # ── Bytes ─────────────────────────────────────────────────────────────────
    fwd_bytes = g(A, "Total Length of Fwd Packet")
    bwd_bytes = g(A, "Total Length of Bwd Packet") + g(B, "Total Length of Fwd Packet") \
              + g(B, "Total Length of Bwd Packet")

    # ── Duración e IAT totales ────────────────────────────────────────────────
    flow_dur  = max(g(A, "Flow Duration"), g(B, "Flow Duration"))
    fwd_iat_t = g(A, "Fwd IAT Total")   # los IATs Fwd pertenecen al subflujo A
    bwd_iat_t = max(g(A, "Bwd IAT Total"), g(B, "Fwd IAT Total") + g(B, "Bwd IAT Total"))

    # ── Headers ──────────────────────────────────────────────────────────────
    fwd_hdr = g(A, "Fwd Header Length")
    bwd_hdr = g(A, "Bwd Header Length") + g(B, "Fwd Header Length") + g(B, "Bwd Header Length")

    # ── Flags ─────────────────────────────────────────────────────────────────
    flag_cols = ["FIN Flag Count", "SYN Flag Count", "RST Flag Count",
                 "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
                 "CWR Flag Count", "ECE Flag Count",
                 "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags"]

    # ── Extremos de longitud de paquete ──────────────────────────────────────
    fwd_len_max = max(g(A, "Fwd Packet Length Max"), g(B, "Bwd Packet Length Max"))
    bwd_len_max = max(g(A, "Bwd Packet Length Max"), g(B, "Fwd Packet Length Max"))
    fwd_len_min = max(0, min(
        g(A, "Fwd Packet Length Min") or np.inf,
        g(B, "Bwd Packet Length Min") or np.inf,
    ))
    if fwd_len_min == np.inf:
        fwd_len_min = 0.0
    bwd_len_min = max(0, min(
        g(A, "Bwd Packet Length Min") or np.inf,
        g(B, "Fwd Packet Length Min") or np.inf,
    ))
    if bwd_len_min == np.inf:
        bwd_len_min = 0.0

    pkt_len_max = max(g(A, "Packet Length Max"), g(B, "Packet Length Max"))
    pkt_len_min = min(
        g(A, "Packet Length Min") or np.inf,
        g(B, "Packet Length Min") or np.inf,
    )
    if pkt_len_min == np.inf:
        pkt_len_min = 0.0

    # ── IAT extremos ──────────────────────────────────────────────────────────
    flow_iat_max = max(g(A, "Flow IAT Max"), g(B, "Flow IAT Max"))
    flow_iat_min = min(
        g(A, "Flow IAT Min") if g(A, "Flow IAT Min") >= 0 else np.inf,
        g(B, "Flow IAT Min") if g(B, "Flow IAT Min") >= 0 else np.inf,
    )
    if flow_iat_min == np.inf:
        flow_iat_min = 0.0

    fwd_iat_max = g(A, "Fwd IAT Max")
    fwd_iat_min = g(A, "Fwd IAT Min")
    bwd_iat_max = max(g(A, "Bwd IAT Max"), g(B, "Fwd IAT Max"))
    bwd_iat_min = min(
        g(A, "Bwd IAT Min") if g(A, "Bwd IAT Min") >= 0 else np.inf,
        g(B, "Fwd IAT Min") if g(B, "Fwd IAT Min") >= 0 else np.inf,
    )
    if bwd_iat_min == np.inf:
        bwd_iat_min = 0.0

    # ── Features derivadas (recalculadas desde conteos) ───────────────────────
    dur_s         = max(flow_dur / 1e6, 1e-9)
    total_pkts    = fwd_total + bwd_total
    total_bytes   = fwd_bytes + bwd_bytes

    flow_bytes_s  = total_bytes / dur_s
    flow_pkts_s   = total_pkts / dur_s
    fwd_pkts_s    = fwd_total  / dur_s
    bwd_pkts_s    = bwd_total  / dur_s
    down_up       = bwd_total  / fwd_total if fwd_total > 0 else 0.0
    avg_pkt_size  = total_bytes / total_pkts if total_pkts > 0 else 0.0

    fwd_len_mean  = fwd_bytes / fwd_total if fwd_total > 0 else 0.0
    bwd_len_mean  = bwd_bytes / bwd_total if bwd_total > 0 else 0.0
    pkt_len_mean  = total_bytes / total_pkts if total_pkts > 0 else 0.0

    fwd_iat_mean  = fwd_iat_t / (fwd_total - 1) if fwd_total > 1 else 0.0
    bwd_iat_mean  = bwd_iat_t / (bwd_total - 1) if bwd_total > 1 else 0.0
    flow_iat_mean = (fwd_iat_t + bwd_iat_t) / (total_pkts - 1) if total_pkts > 1 else 0.0

    fwd_act_data  = int(g(A, "Fwd Act Data Pkts"))

    # ── Stds y varianza: combinadas con fórmula exacta ────────────────────────
    # Fwd: solo subflujo A (los Fwd de B son realmente Bwd)
    fwd_len_std = float(g(A, "Fwd Packet Length Std"))

    # Bwd: combinar Bwd de A con Fwd de B (que son Bwd reales del flujo original)
    bwd_len_std = _std_ponderada(
        g(A, "Bwd Packet Length Mean"), g(A, "Bwd Packet Length Std"), bwd_A,
        g(B, "Fwd Packet Length Mean"), g(B, "Fwd Packet Length Std"), fwd_B,
    )

    pkt_len_std = _std_ponderada(
        g(A, "Packet Length Mean"), g(A, "Packet Length Std"), int(g(A, "Total Fwd Packet")) + int(g(A, "Total Bwd packets")),
        g(B, "Packet Length Mean"), g(B, "Packet Length Std"), int(g(B, "Total Fwd Packet")) + int(g(B, "Total Bwd packets")),
    )
    pkt_len_var = pkt_len_std ** 2

    fwd_iat_std = float(g(A, "Fwd IAT Std"))
    bwd_iat_std = _std_ponderada(
        g(A, "Bwd IAT Mean"), g(A, "Bwd IAT Std"), max(bwd_A - 1, 0),
        g(B, "Fwd IAT Mean"), g(B, "Fwd IAT Std"), max(fwd_B - 1, 0),
    )
    n_flow_intervals = total_pkts - 1
    flow_iat_std = _std_ponderada(
        g(A, "Flow IAT Mean"), g(A, "Flow IAT Std"), max(int(g(A,"Total Fwd Packet"))+int(g(A,"Total Bwd packets"))-1, 0),
        g(B, "Flow IAT Mean"), g(B, "Flow IAT Std"), max(int(g(B,"Total Fwd Packet"))+int(g(B,"Total Bwd packets"))-1, 0),
    )

    # ── Init Win Bytes: del primer paquete del subflujo correcto ──────────────
    init_win_fwd = int(g(A, "FWD Init Win Bytes"))   # SYN desde atacante → subflujo A
    init_win_bwd = int(g(A, "Bwd Init Win Bytes"))    # SYN-ACK desde victima → subflujo A
    if init_win_bwd == 0:
        init_win_bwd = int(g(B, "FWD Init Win Bytes"))  # fallback al subflujo B

    # ── Construir la fila reconstruida ────────────────────────────────────────
    row = A.copy()   # heredar metadata (Archivo_Origen, _ts, Label, etc.)

    row["Total Fwd Packet"]             = fwd_total
    row["Total Bwd packets"]            = bwd_total
    row["Total Length of Fwd Packet"]   = fwd_bytes
    row["Total Length of Bwd Packet"]   = bwd_bytes
    row["Flow Duration"]                = flow_dur
    row["Fwd IAT Total"]                = fwd_iat_t
    row["Bwd IAT Total"]                = bwd_iat_t
    row["Fwd Header Length"]            = fwd_hdr
    row["Bwd Header Length"]            = bwd_hdr

    row["Fwd Packet Length Max"]        = fwd_len_max
    row["Fwd Packet Length Min"]        = fwd_len_min
    row["Fwd Packet Length Mean"]       = fwd_len_mean
    row["Fwd Packet Length Std"]        = fwd_len_std
    row["Fwd Segment Size Avg"]         = fwd_len_mean

    row["Bwd Packet Length Max"]        = bwd_len_max
    row["Bwd Packet Length Min"]        = bwd_len_min
    row["Bwd Packet Length Mean"]       = bwd_len_mean
    row["Bwd Packet Length Std"]        = bwd_len_std
    row["Bwd Segment Size Avg"]         = bwd_len_mean

    row["Packet Length Min"]            = pkt_len_min
    row["Packet Length Max"]            = pkt_len_max
    row["Packet Length Mean"]           = pkt_len_mean
    row["Packet Length Std"]            = pkt_len_std
    row["Packet Length Variance"]       = pkt_len_var

    row["Flow Bytes/s"]                 = flow_bytes_s
    row["Flow Packets/s"]               = flow_pkts_s
    row["Fwd Packets/s"]                = fwd_pkts_s
    row["Bwd Packets/s"]                = bwd_pkts_s
    row["Down/Up Ratio"]                = down_up
    row["Average Packet Size"]          = avg_pkt_size

    row["Flow IAT Mean"]                = flow_iat_mean
    row["Flow IAT Std"]                 = flow_iat_std
    row["Flow IAT Max"]                 = flow_iat_max
    row["Flow IAT Min"]                 = flow_iat_min

    row["Fwd IAT Mean"]                 = fwd_iat_mean
    row["Fwd IAT Std"]                  = fwd_iat_std
    row["Fwd IAT Max"]                  = fwd_iat_max
    row["Fwd IAT Min"]                  = fwd_iat_min

    row["Bwd IAT Mean"]                 = bwd_iat_mean
    row["Bwd IAT Std"]                  = bwd_iat_std
    row["Bwd IAT Max"]                  = bwd_iat_max
    row["Bwd IAT Min"]                  = bwd_iat_min

    row["FWD Init Win Bytes"]           = init_win_fwd
    row["Bwd Init Win Bytes"]           = init_win_bwd
    row["Fwd Act Data Pkts"]            = fwd_act_data

    for col in flag_cols:
        if col in A.index and col in B.index:
            row[col] = int(g(A, col)) + int(g(B, col))

    return row


# ── Argumentos ────────────────────────────────────────────────────────────────
csv_old = sys.argv[1] if len(sys.argv) > 1 else "ataques_adversarios_COMPLETOS.csv"
csv_new = sys.argv[2] if len(sys.argv) > 2 else "REPORTE_GLOBAL_PREDICCIONES.csv"
output  = sys.argv[3] if len(sys.argv) > 3 else "heatmap_dos_csv.pdf"

print(f"OLD: {csv_old}")
print(f"NEW: {csv_new}")

df_old_raw = pd.read_csv(csv_old)
df_new_raw = pd.read_csv(csv_new)

print(f"  Filas OLD (raw): {len(df_old_raw)}  |  Filas NEW: {len(df_new_raw)}")

# ── Reproducir el filtro del generador sobre OLD ──────────────────────────────
mask_ok = (
    (df_old_raw["Init_Win_bytes_forward"] > 0)
    & (df_old_raw["Flow Packets/s"] < 1e9)
    & ~((df_old_raw["Flow Duration"] == 0) & (df_old_raw["Total Fwd Packets"] > 1))
    & (df_old_raw["Flow IAT Min"] >= 0)
    & (df_old_raw["Total Fwd Packets"] > 0)
)
df_old = df_old_raw[mask_ok].copy().reset_index(drop=True)
print(f"  Filas OLD tras filtro generador: {len(df_old)}")

# ── Extraer timestamp del nombre de PCAP ─────────────────────────────────────
def _extraer_ts(nombre: str):
    m = re.search(r"_(\d{10})_ISCX\.csv$", str(nombre))
    return int(m.group(1)) if m else None

df_new_raw["_ts"] = df_new_raw["Archivo_Origen"].apply(_extraer_ts)
n_sin_ts = df_new_raw["_ts"].isna().sum()
if n_sin_ts:
    print(f"  ⚠  {n_sin_ts} filas sin timestamp extraíble — se descartan")
    df_new_raw = df_new_raw[df_new_raw["_ts"].notna()].copy()

# ── Reconstruir PCAPs partidos / colapsar los de 1 solo flujo ─────────────────
filas_reconstruidas = []
n_splits = 0

for ts, grupo in df_new_raw.groupby("_ts", sort=False):
    if len(grupo) == 1:
        filas_reconstruidas.append(grupo.iloc[0])
    else:
        n_splits += 1
        filas_reconstruidas.append(reconstruir_flujo(grupo))

df_new = pd.DataFrame(filas_reconstruidas).sort_values("_ts").reset_index(drop=True)
print(f"  {n_splits} PCAPs reconstruidos (subflujos agregados correctamente)")
print(f"  Filas NEW tras reconstrucción: {len(df_new)}")

# ── Join exacto por timestamp ordinal ─────────────────────────────────────────
n_pcap = len(df_new)
n_old  = len(df_old)

if n_pcap != n_old:
    print(f"  ⚠  Nº PCAPs ({n_pcap}) ≠ filas OLD filtradas ({n_old}). "
          f"Usando los primeros {min(n_pcap, n_old)} pares.")
    n = min(n_pcap, n_old)
    df_new = df_new.iloc[:n].reset_index(drop=True)
    df_old = df_old.iloc[:n].reset_index(drop=True)
else:
    print(f"  ✓  Join exacto: {n_pcap} pares (timestamp ordinal ↔ fila OLD filtrada)")

n_rows = len(df_new)

# ── Filtrar features válidas ──────────────────────────────────────────────────
valid_features = []
missing = []
for f in FEATURES:
    col_old = f
    col_new = COL_MAP.get(f, f)
    if col_old in df_old.columns and col_new in df_new.columns:
        valid_features.append(f)
    else:
        missing.append(f"'{f}' → OLD:{'OK' if col_old in df_old.columns else 'FALTA'}  "
                       f"NEW('{col_new}'):{'OK' if col_new in df_new.columns else 'FALTA'}")

print(f"  {len(valid_features)} features válidas para el heatmap")
if missing:
    print("  Features descartadas:")
    for m in missing:
        print(f"    {m}")

# ── Matriz de similitud ───────────────────────────────────────────────────────
n = len(valid_features)
sim_matrix = np.full((n, n), np.nan)

for i, fi in enumerate(valid_features):
    col_new_i = COL_MAP.get(fi, fi)
    nv = pd.to_numeric(df_new[col_new_i], errors="coerce").values

    for j, fj in enumerate(valid_features):
        o = pd.to_numeric(df_old[fj], errors="coerce").values

        mask = ~np.isnan(nv) & ~np.isnan(o)
        a = nv[mask]
        b = o[mask]
        if len(a) < 2:
            continue

        denom = np.maximum(np.abs(a), np.abs(b)) + 1e-9
        ratio = np.clip(np.abs(a - b) / denom, 0, 1.0)
        sim_matrix[i, j] = 1.0 - ratio.mean()

print(f"\nRango similitud: min={np.nanmin(sim_matrix):.4f}  max={np.nanmax(sim_matrix):.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
cell     = 0.5
margin_l = 4.0
margin_b = 4.0
fig_w    = margin_l + n * cell
fig_h    = margin_b + n * cell

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.patch.set_facecolor("#ffffff")
ax.set_facecolor("#ffffff")

im = ax.imshow(sim_matrix, aspect="auto", cmap="Blues",
               vmin=0, vmax=1, interpolation="nearest", origin="upper")

tick_colors = ["#b71c1c" if f in GAN_MODIFIED else "#1a2a3a" for f in valid_features]

ax.set_xticks(range(n))
ax.set_xticklabels([f[:20] + "…" if len(f) > 20 else f for f in valid_features],
                   rotation=90, fontsize=8, ha="right")
for tick, col in zip(ax.get_xticklabels(), tick_colors):
    tick.set_color(col)
    if col == "#b71c1c":
        tick.set_fontweight("bold")

ax.set_yticks(range(n))
ax.set_yticklabels([f[:22] + "…" if len(f) > 22 else f for f in valid_features], fontsize=8)
for tick, col in zip(ax.get_yticklabels(), tick_colors):
    tick.set_color(col)
    if col == "#b71c1c":
        tick.set_fontweight("bold")

ax.set_xlabel(f"OLD: {csv_old.split('/')[-1]}", fontsize=11,
              color="#0d47a1", labelpad=15, fontweight="bold")
ax.set_ylabel(f"NEW: {csv_new.split('/')[-1]}", fontsize=11,
              color="#0d47a1", labelpad=15, fontweight="bold")
ax.tick_params(axis="both", which="both", length=0, pad=5)
for sp in ax.spines.values():
    sp.set_visible(False)

for x in range(n):
    ax.axvline(x - 0.5, color="#e0e8f0", linewidth=0.5)
for y in range(n):
    ax.axhline(y - 0.5, color="#e0e8f0", linewidth=0.5)

for i, f in enumerate(valid_features):
    color = "#c62828" if f in GAN_MODIFIED else "#37474f"
    lw    = 2.0       if f in GAN_MODIFIED else 0.8
    ax.add_patch(mpatches.Rectangle((i - 0.5, i - 0.5), 1, 1,
                 fill=False, edgecolor=color, linewidth=lw, zorder=3))

cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.015, shrink=0.75)
cbar.set_label("Similitud fila a fila  (0=diferente · 1=idéntico)",
               fontsize=10, color="#1a2a3a", labelpad=10)
cbar.ax.tick_params(labelsize=9, colors="#1a2a3a")
cbar.outline.set_edgecolor("#90aac8")

legend_elements = [
    mpatches.Patch(facecolor="none", edgecolor="#c62828", linewidth=2.0,
                   label="Modificada por GAN (LIMITES_FISICOS)"),
    mpatches.Patch(facecolor="none", edgecolor="#37474f", linewidth=0.8,
                   label="Sin restricción explícita"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=9,
          framealpha=0.9, edgecolor="#aabbcc", bbox_to_anchor=(1.0, 1.01))

ax.set_title(
    "Mapa de calor de similitud  OLD vs NEW\n",
    fontsize=13, pad=20, fontweight="bold"
)

plt.tight_layout()
plt.savefig(output, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\n[OK] Heatmap guardado en: {output}")

# ── Diagnóstico diagonal ──────────────────────────────────────────────────────
print(f"\nDiagonal — Similitud OLD_X vs NEW_X (lo que más importa):")
print(f"  {'Feature':<38} {'Similitud':>10}  {'GAN':>6}")
print("  " + "-" * 60)
for i, f in enumerate(valid_features):
    v     = sim_matrix[i, i]
    marca = "◀ SÍ" if f in GAN_MODIFIED else ""
    flag  = "⚠"    if f in GAN_MODIFIED and v > 0.85 else ""
    print(f"  {f:<38} {v:>10.4f}  {marca:>6} {flag}")