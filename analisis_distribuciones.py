import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

RUTA_DATASET = r"Dataset/MachineLearningCVE"

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
# PALETA DE 15 COLORES PERCEPTUALMENTE DISTINTOS
# Cada color es distinguible incluso en impresión en escala de grises
# gracias al refuerzo con patrones de trama (hatch)
# ==========================================

# El orden importa: el índice 0 siempre será BENIGN
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

# Hatch diferente por clase: BENIGN sin hatch, ataques con hatch creciente
# Esto permite distinguirlos incluso si dos colores parecen similares en pantalla
HATCHES = [
    None,    # 0  BENIGN: sin trama
    '////',  # 1
    '\\\\',  # 2
    '||||',  # 3
    '----',  # 4
    '++++',  # 5
    'xxxx',  # 6
    '....',  # 7
    '****',  # 8
    'oooo',  # 9
    'OO',    # 10
    '//',    # 11
    '\\',    # 12
    '||',    # 13
    '--',    # 14
]

# ==========================================
# FUNCIONES
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
        # float32 → mitad de RAM, lectura más rápida
        dtype_map = {c: 'float32' for c in columnas_a_cargar if c != label_real}
        df = pd.read_csv(archivo, usecols=columnas_a_cargar, dtype=dtype_map)
        rename_map = {v: k for k, v in col_map.items()}
        df = df.rename(columns=rename_map)
        return df

    except Exception as e:
        print(f"    [-] Error leyendo {os.path.basename(archivo)}: {e}")
        return None


def asignar_estilos(etiquetas_ordenadas):
    """
    Devuelve un dict {etiqueta: (color, hatch)} garantizando que
    BENIGN siempre recibe el índice 0 y el resto se asigna en orden.
    """
    estilos = {}
    idx_ataque = 1  # los ataques empiezan en el índice 1 de la paleta
    for etiqueta in etiquetas_ordenadas:
        if 'BENIGN' in etiqueta.upper():
            estilos[etiqueta] = (PALETTE[0], HATCHES[0])
        else:
            i = idx_ataque % len(PALETTE)
            estilos[etiqueta] = (PALETTE[i], HATCHES[i])
            idx_ataque += 1
    return estilos


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
    print(f"[*] Total flujos cargados: {len(df_total):,}")

    cols_a_graficar = [c for c in FEATURES_TO_PLOT if c in df_total.columns]

    # Orden fijo de etiquetas: BENIGN primero, luego ataques por frecuencia desc.
    # Así los colores son CONSISTENTES entre todos los gráficos.
    conteo_global = df_total['Label'].value_counts()
    etiquetas_ordenadas = (
        [e for e in conteo_global.index if 'BENIGN' in e.upper()] +
        [e for e in conteo_global.index if 'BENIGN' not in e.upper()]
    )
    estilos = asignar_estilos(etiquetas_ordenadas)

    for feature_name in cols_a_graficar:
        print(f"\n[*] Generando gráfico para: {feature_name}...")

        df_clean = df_total[[feature_name, 'Label']].replace([np.inf, -np.inf], np.nan).dropna()
        all_values = df_clean[feature_name]
        if len(all_values) == 0:
            continue

        max_real = all_values.max()
        limite_superior = np.percentile(all_values, 99)
        if limite_superior <= 0:
            limite_superior = max_real
        if limite_superior <= 0:
            limite_superior = 1

        print(f"    -> Rango visual: 0 - {limite_superior:.2f}")

        bins_fijos = np.linspace(0, limite_superior, 50)

        # Ordenamos de menor a mayor frecuencia para que las clases pequeñas
        # (ataques raros) queden encima y no queden enterradas
        conteo_clases = []
        for etiqueta in etiquetas_ordenadas:
            data = df_clean[df_clean['Label'] == etiqueta][feature_name]
            if len(data) > 0:
                conteo_clases.append((etiqueta, data, len(data)))
        conteo_clases.sort(key=lambda x: x[2])  # menor frecuencia → se dibuja última (encima)

        fig, ax = plt.subplots(figsize=(14, 7))

        for etiqueta, data, count in conteo_clases:
            data_visible = data[data <= limite_superior]
            # Normalización: proporción respecto al TOTAL de esa clase
            # (no respecto a los visibles) → comparable entre clases
            weights = np.ones_like(data_visible) / len(data)

            color, hatch = estilos.get(etiqueta, (PALETTE[1], HATCHES[1]))

            ax.hist(
                data_visible,
                bins=bins_fijos,
                weights=weights,
                alpha=0.55,
                label=etiqueta,
                color=color,
                hatch=hatch,
                edgecolor='black',
                linewidth=0.6,
            )

        # ── Título y ejes ────────────────────────────────────────────────────
        ax.set_title(
            f"Distribución Relativa (Lineal): {feature_name}",
            fontsize=15, fontweight='bold', pad=12
        )
        info_max = f"(Max visible: {limite_superior:.0f} / Max real: {max_real:.0f})"
        ax.set_xlabel(f"Valor Real ({feature_name})  {info_max}", fontsize=11)
        ax.set_ylabel("Frecuencia Relativa (Proporción por clase)", fontsize=11)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:g}'))

        # ── Leyenda en 2 columnas fuera del área del gráfico ─────────────────
        # Así nunca tapa las barras
        handles, labels_leg = ax.get_legend_handles_labels()
        # Invertimos para que BENIGN quede arriba
        ax.legend(
            list(reversed(handles)),
            list(reversed(labels_leg)),
            title="Tipo de Tráfico",
            loc='upper left',
            bbox_to_anchor=(1.01, 1),  # fuera del plot, a la derecha
            borderaxespad=0,
            ncol=1,
            fontsize=8.5,
            title_fontsize=9.5,
            framealpha=0.95,
        )

        ax.grid(True, which='both', ls='--', alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)

        filename = f"grafico_lineal_{feature_name.replace(' ', '_')}.png"
        fig.tight_layout()
        fig.savefig(filename, dpi=200, bbox_inches='tight')
        print(f"    [✓] Guardado: {filename}")
        plt.close(fig)

    print("\n[✓] ¡Todos los gráficos generados!")


if __name__ == "__main__":
    generar_graficos_lineales(RUTA_DATASET)