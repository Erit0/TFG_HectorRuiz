import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import glob
import os
import numpy as np

# ==========================================
# CONFIGURACIÓN
# ==========================================

FEATURES_TO_PLOT = [
    'Init_Win_bytes_forward',
    'Init_Win_bytes_backward',
    'Bwd Packet Length Min',
    'Fwd Header Length',
    'Bwd Packet Length Std',
    'Bwd Packet Length Mean',
    'Avg Bwd Segment Size',
    'Bwd Packet Length Max',
    'Packet Length Mean',
    'Max Packet Length',
    'Total Length of Bwd Packets',
    'Flow IAT Mean'
]

RUTA_DATASET = r"Para_generar_graficos"
#RUTA_DATASET = r"Dataset/MachineLearningCVE"

UNITS_MAP = {
    'Init_Win_bytes_forward': '(Bytes)',
    'Init_Win_bytes_backward': '(Bytes)',
    'Bwd Packet Length Min': '(Bytes)',
    'Fwd Header Length': '(Bytes)',
    'Bwd Packet Length Std': '(Bytes)',
    'Bwd Packet Length Mean': '(Bytes)',
    'Avg Bwd Segment Size': '(Bytes)',
    'Bwd Packet Length Max': '(Bytes)',
    'Packet Length Mean': '(Bytes)',
    'Max Packet Length': '(Bytes)',
    'Total Length of Bwd Packets': '(Bytes)',
    'Flow IAT Mean': '(Microsegundos)'
}

# ==========================================
# PALETA DE 15 COLORES
# ==========================================

PALETTE = [
    '#1f77b4',  # 0  azul acero        → BENIGN
    '#d62728',  # 1  rojo              → ataque 1
    '#2ca02c',  # 2  verde bosque      → ataque 2
    '#ff7f0e',  # 3  naranja           → ataque 3
    '#9467bd',  # 4  violeta           → ataque 4
    '#17becf',  # 5  cian             → ataque 5
    '#e377c2',  # 6  rosa             → ataque 6
    '#8c564b',  # 7  marrón           → ataque 7
    '#bcbd22',  # 8  amarillo-verde   → ataque 8
    '#7f7f7f',  # 9  gris medio       → ataque 9
    '#1a1a6e',  # 10 azul marino      → ataque 10
    '#e7ba52',  # 11 dorado           → ataque 11
    '#006d2c',  # 12 verde oscuro     → ataque 12
    '#e6550d',  # 13 naranja tostado  → ataque 13
    '#54278f',  # 14 púrpura oscuro   → ataque 14
]

# Pool de patrones disponibles para asignar cuando haga falta
HATCH_POOL = [
    '////',      # diagonales densas
    '\\\\\\\\',  # diagonales inversas densas
    '||||',      # verticales
    '----',      # horizontales
    '++++',      # cruz
    'xxxx',      # aspa
    '....',      # puntos
    '****',      # asteriscos
    'oooo',      # círculos
    'OO',        # círculos grandes
]

# Umbral de distancia perceptual CIELAB por debajo del cual
# dos colores se consideran "demasiado similares" y necesitan hatch.
# Delta-E < 25 → similar; > 25 → claramente distinto.
UMBRAL_DELTA_E = 25.0

# ==========================================
# LÍMITES FIJOS DEL EJE X
# ==========================================
XLIM_FIJOS = {
    'Init_Win_bytes_forward':      65535.00,
    'Init_Win_bytes_backward':     42780.00,
    'Bwd Packet Length Min':         248.00,
    'Fwd Header Length':            1328.00,
    'Bwd Packet Length Std':        4128.32,
    'Bwd Packet Length Mean':       2321.40,
    'Avg Bwd Segment Size':         2321.40,
    'Bwd Packet Length Max':        8760.00,
    'Packet Length Mean':           1292.56,
    'Max Packet Length':           10135.00,
    'Total Length of Bwd Packets': 71831.00,
    'Flow IAT Mean':            21400000.00,
}

# ==========================================
# CONVERSIÓN DE COLOR Y DISTANCIA PERCEPTUAL
# ==========================================

def hex_to_rgb01(hex_color):
    """Convierte '#rrggbb' a (r, g, b) en rango [0, 1]."""
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def rgb_to_lab(rgb):
    """
    Convierte RGB [0,1] → CIE L*a*b* pasando por XYZ con iluminante D65.
    Permite calcular distancias perceptuales reales (Delta-E).
    """
    # Paso 1: linearizar (deshacer gamma sRGB)
    def linearize(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = [linearize(c) for c in rgb]

    # Paso 2: RGB lineal → XYZ (matriz sRGB D65)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # Paso 3: normalizar por iluminante D65
    x /= 0.95047
    y /= 1.00000
    z /= 1.08883

    # Paso 4: XYZ → L*a*b*
    def f(t):
        return t ** (1/3) if t > 0.008856 else 7.787 * t + 16/116

    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_star = 200 * (fy - fz)
    return np.array([L, a, b_star])


def delta_e(hex1, hex2):
    """Delta-E (CIE76) entre dos colores hex. < 1 = imperceptible, > 25 = claramente distinto."""
    lab1 = rgb_to_lab(hex_to_rgb01(hex1))
    lab2 = rgb_to_lab(hex_to_rgb01(hex2))
    return np.linalg.norm(lab1 - lab2)


def calcular_hatches_necesarios(colores_activos, umbral=UMBRAL_DELTA_E):
    """
    Dado un dict {etiqueta: color_hex} con los colores que aparecen en
    el gráfico actual, determina qué etiquetas necesitan hatch porque
    su color es demasiado similar al de alguna otra etiqueta presente,
    o porque están en la lista CLASES_HATCH_FORZADO.

    Devuelve un dict {etiqueta: hatch_string_o_''}.
    """
    # Clases que siempre llevan hatch, independientemente del color
    CLASES_HATCH_FORZADO = {
        'PortScan',
    }

    etiquetas = list(colores_activos.keys())
    necesita_hatch = set()

    # Forzar hatch en las clases de la lista (búsqueda case-insensitive)
    for et in etiquetas:
        if any(et.strip().lower() == forzada.lower() for forzada in CLASES_HATCH_FORZADO):
            necesita_hatch.add(et)

    # Añadir clases cuyos colores son perceptualmente similares a otro
    for i in range(len(etiquetas)):
        for j in range(i + 1, len(etiquetas)):
            et_i = etiquetas[i]
            et_j = etiquetas[j]
            dist = delta_e(colores_activos[et_i], colores_activos[et_j])
            if dist < umbral:
                necesita_hatch.add(et_i)
                necesita_hatch.add(et_j)

    # Asignar un patrón distinto a cada clase que lo necesita
    resultado = {}
    hatch_idx = 0
    for etiqueta in etiquetas:
        if etiqueta in necesita_hatch:
            resultado[etiqueta] = HATCH_POOL[hatch_idx % len(HATCH_POOL)]
            hatch_idx += 1
        else:
            resultado[etiqueta] = ''   # sin hatch → barra limpia

    return resultado


# ==========================================
# FUNCIONES PRINCIPALES
# ==========================================

def cargar_archivo_seguro(archivo, columnas_deseadas):
    try:
        df_header = pd.read_csv(archivo, nrows=0)
        col_map = {}

        label_real = next((c for c in df_header.columns if c.strip().lower() == 'label'), None)
        if not label_real:
            return None
        col_map['Label'] = label_real

        for feature in columnas_deseadas:
            feature_clean = feature.lower().replace(' ', '').replace('_', '').replace('/', '')
            for col_real in df_header.columns:
                col_real_clean = col_real.strip().lower().replace(' ', '').replace('_', '').replace('/', '')
                if feature_clean == col_real_clean:
                    col_map[feature] = col_real
                    break
                if 'header' in feature_clean and 'len' in col_real_clean:
                    if feature_clean.replace('length', 'len') == col_real_clean:
                        col_map[feature] = col_real
                        break

        if len(col_map) < 2:
            return None

        columnas_a_cargar = list(col_map.values())
        dtype_map = {c: 'float32' for c in columnas_a_cargar if c != label_real}
        df = pd.read_csv(archivo, usecols=columnas_a_cargar, dtype=dtype_map)
        rename_map = {v: k for k, v in col_map.items()}
        df = df.rename(columns=rename_map)
        return df

    except Exception as e:
        print(f"    [-] Error leyendo {os.path.basename(archivo)}: {e}")
        return None


def asignar_colores(etiquetas_ordenadas):
    """Color fijo por etiqueta. BENIGN siempre azul acero (índice 0)."""
    colores = {}
    idx_ataque = 1
    for etiqueta in etiquetas_ordenadas:
        if 'BENIGN' in etiqueta.upper():
            colores[etiqueta] = PALETTE[0]
        else:
            colores[etiqueta] = PALETTE[idx_ataque % len(PALETTE)]
            idx_ataque += 1
    return colores


def calcular_histograma(data, bins, total_count):
    counts, _ = np.histogram(data, bins=bins)
    return counts / total_count


def generar_graficos_lineales(ruta_directorio):
    print(f"[*] Buscando archivos CSV en: {ruta_directorio}...")
    archivos_csv = glob.glob(os.path.join(ruta_directorio, "*.csv"))

    if not archivos_csv:
        print("[-] Error: No se encontraron archivos .csv.")
        return

    print("[*] Cargando dataset...")
    lista_dfs = []
    for archivo in archivos_csv:
        df = cargar_archivo_seguro(archivo, FEATURES_TO_PLOT)
        if df is not None:
            lista_dfs.append(df)

    if not lista_dfs:
        print("[-] No se pudieron cargar datos.")
        return

    df_total = pd.concat(lista_dfs, ignore_index=True)
    df_total['Label'] = df_total['Label'].str.encode('ascii', 'ignore').str.decode('ascii').str.strip()
    df_total['Label'] = df_total['Label'].str.replace(r'\s+', ' ', regex=True)

    print(f"[*] Total flujos cargados: {len(df_total):,}")

    cols_a_graficar = [c for c in FEATURES_TO_PLOT if c in df_total.columns]

    conteo_global = df_total['Label'].value_counts()
    etiquetas_ordenadas = (
        [e for e in conteo_global.index if 'BENIGN' in e.upper()] +
        [e for e in conteo_global.index if 'BENIGN' not in e.upper()]
    )

    # Colores fijos para todo el script (consistencia entre gráficos)
    colores = asignar_colores(etiquetas_ordenadas)

    # ── Pre-calcular qué clases necesitan hatch (igual para todos los gráficos)
    # Usamos todos los colores del dataset, no solo los de cada gráfico,
    # para que los patrones sean también consistentes entre PDFs.
    hatches = calcular_hatches_necesarios(colores)

    print("\n[*] Clases con hatch asignado (colores similares detectados):")
    for et, h in hatches.items():
        estado = f"hatch='{h}'" if h else "sin hatch"
        print(f"    {et}: {estado}")

    plt.rcParams['hatch.linewidth'] = 1.5

    for feature_name in cols_a_graficar:
        print(f"\n[*] Generando gráfico para: {feature_name}...")

        df_clean = df_total[[feature_name, 'Label']].replace([np.inf, -np.inf], np.nan).dropna()
        all_values = df_clean[feature_name]
        if len(all_values) == 0:
            continue

        max_real = all_values.max()
        if feature_name in XLIM_FIJOS:
            limite_superior = XLIM_FIJOS[feature_name]
        else:
            limite_superior = np.percentile(all_values, 99)
        if limite_superior <= 0:
            limite_superior = max_real
        if limite_superior <= 0:
            limite_superior = 1

        print(f"    -> Rango visual: 0 - {limite_superior:.2f}")

        bins_fijos  = np.linspace(0, limite_superior, 50)
        bin_width   = bins_fijos[1] - bins_fijos[0]
        bin_centers = (bins_fijos[:-1] + bins_fijos[1:]) / 2

        # ── Calcular histogramas ──────────────────────────────────────────
        clases_info = []
        for etiqueta in etiquetas_ordenadas:
            data         = df_clean[df_clean['Label'] == etiqueta][feature_name]
            data_visible = data[data <= limite_superior]
            if len(data_visible) == 0:
                continue
            alturas    = calcular_histograma(data_visible, bins_fijos, len(data))
            altura_max = alturas.max()
            clases_info.append({
                'etiqueta':   etiqueta,
                'alturas':    alturas,
                'altura_max': altura_max,
            })

        if not clases_info:
            continue

        # Mayor altura_max primero → se dibuja detrás
        clases_info.sort(key=lambda x: x['altura_max'], reverse=True)

        fig, ax = plt.subplots(figsize=(14, 7))

        for clase in clases_info:
            etiqueta = clase['etiqueta']
            alturas  = clase['alturas']
            color    = colores[etiqueta]
            hatch    = hatches.get(etiqueta, '')

            # Capa 1: relleno sólido con el color de la clase
            ax.bar(
                bin_centers,
                alturas,
                width=bin_width,
                color=color,
                edgecolor='none',
                linewidth=0,
            )

            # Capa 2: hatch negro (solo si esta clase lo necesita)
            if hatch:
                ax.bar(
                    bin_centers,
                    alturas,
                    width=bin_width,
                    facecolor='none',
                    edgecolor='black',
                    linewidth=0.5,
                    hatch=hatch,
                )

        # ── Leyenda manual con mpatches ───────────────────────────────────
        legend_handles = []
        for etiqueta in etiquetas_ordenadas:
            if not any(c['etiqueta'] == etiqueta for c in clases_info):
                continue
            color = colores[etiqueta]
            hatch = hatches.get(etiqueta, '')
            patch = mpatches.Patch(
                facecolor=color,
                edgecolor='black' if hatch else color,
                hatch=hatch,
                label=etiqueta,
                linewidth=0.5,
            )
            legend_handles.append(patch)

        ax.legend(
            handles=legend_handles,
            title="Tipo de Tráfico",
            loc='upper left',
            bbox_to_anchor=(1.01, 1),
            borderaxespad=0,
            ncol=1,
            fontsize=8.5,
            title_fontsize=9.5,
            framealpha=0.95,
        )

        # ── Título y ejes ─────────────────────────────────────────────────
        ax.set_title(
            f"Distribución Relativa (Lineal): {feature_name}",
            fontsize=15, fontweight='bold', pad=12
        )
        info_max = f"(Max visible: {limite_superior:.0f} / Max real: {max_real:.0f})"
        unidad = UNITS_MAP.get(feature_name, '')
        ax.set_xlabel(f"Valor Real ({feature_name}) {unidad}  {info_max}", fontsize=11)
        ax.set_ylabel("Frecuencia Relativa (Proporción por clase)", fontsize=11)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:g}'))

        ax.grid(True, which='both', ls='--', alpha=0.25, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim(0, limite_superior)

        filename = f"grafico_lineal_{feature_name.replace(' ', '_')}.pdf"
        fig.tight_layout()
        fig.savefig(filename, dpi=200, bbox_inches='tight')
        print(f"    [✓] Guardado: {filename}")
        plt.close(fig)

    print("\n[✓] ¡Todos los gráficos generados!")


if __name__ == "__main__":
    generar_graficos_lineales(RUTA_DATASET)