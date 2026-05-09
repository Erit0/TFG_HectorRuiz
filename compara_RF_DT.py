"""
Evaluación y Comparación Visual: Random Forest vs Decision Tree
Dataset: CIC-IDS-2017
Autor: Script para TFG

Genera un PDF con comparativa completa de ambos modelos para incluir en el TFG,
con soporte para tildes, nombres de ataques limpios y métricas en español.
"""

import os
import sys
import codecs
import time
import warnings
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend sin display
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
from glob import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
import joblib

warnings.filterwarnings('ignore')

# ─── CONFIGURACIÓN DE FUENTE CON SOPORTE UNICODE Y TILDES ───────────────────
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

def configurar_fuente_unicode():
    """Busca y establece una fuente del sistema con soporte unicode completo para tildes."""
    fuentes_candidatas = [
        'Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 
        'Verdana', 'Tahoma', 'Noto Sans'
    ]
    fuentes_disponibles = {f.name for f in fm.fontManager.ttflist}
    for fuente in fuentes_candidatas:
        if fuente in fuentes_disponibles:
            matplotlib.rcParams['font.family'] = fuente
            print(f"  Fuente seleccionada para PDF: {fuente} (Soporta tildes)")
            return
    matplotlib.rcParams['font.family'] = 'sans-serif'
    print("  Fuente: sans-serif (fallback)")

configurar_fuente_unicode()
matplotlib.rcParams['axes.unicode_minus'] = False 
matplotlib.rcParams['pdf.fonttype'] = 42 # Fuerza la incrustación de tildes en el PDF

# ─── LIMPIEZA DE NOMBRES DE ATAQUES ──────────────────────────────────────────
def sanitizar_etiqueta(texto):
    """Elimina símbolos raros y formatea nombres para el TFG."""
    # Eliminar caracteres no ASCII (como el símbolo oculto en CIC-IDS-2017)
    texto = texto.encode("ascii", "ignore").decode("ascii")
    # Reemplazar múltiples espacios o guiones bajos por uno solo
    texto = re.sub(r'[_ \t]+', ' ', texto).strip()
    
    # Mapeo manual para mayor legibilidad en el TFG
    mapeo = {
        'BENIGN': 'Benigno',
        'Web Attack Brute Force': 'Web Brute Force',
        'Web Attack XSS': 'Web XSS',
        'Web Attack Sql Injection': 'Web SQL Injection',
        'FTP-Patator': 'FTP Patator',
        'SSH-Patator': 'SSH Patator',
        'DoS Hulk': 'DoS Hulk',
        'DoS GoldenEye': 'DoS GoldenEye',
        'DoS slowloris': 'DoS Slowloris',
        'DoS Slowhttptest': 'DoS SlowHTTP'
    }
    return mapeo.get(texto, texto)

# ─── MISMAS COLUMNAS QUE EL SCRIPT ORIGINAL ──────────────────────────────────
columnas_a_eliminar = [
    'Destination Port', 'Flow ID', 'Source IP', 'Source Port',
    'Destination IP', 'Timestamp',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    'URG Flag Count', 'CWE Flag Count', 'CWR Flag Count', 'ECE Flag Count',
    'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Fwd Avg Bytes/Bulk',
    'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
    'Subflow Fwd Packets', 'Subflow Fwd Bytes',
    'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
    'Fwd Seg Size Min', 'min_seg_size_forward',
    'Fwd Header Length.1'
]

# ─── PALETA DE COLORES PARA EL TFG ───────────────────────────────────────────
COLOR_RF  = '#2563EB'   # Azul
COLOR_DT  = '#DC2626'   # Rojo
COLOR_BG  = '#F8FAFC'
COLOR_GRID = '#E2E8F0'

# ─── CARGA Y PREPROCESADO ─────────────────────────────────────────────────────
def load_cic_dataset(path_pattern="Dataset/MachineLearningCVE/*.csv"):
    files = glob(path_pattern)
    if not files:
        raise FileNotFoundError(f"No CSV files found for pattern: {path_pattern}")
    df_list = [pd.read_csv(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    df.columns = df.columns.str.strip()
    return df

def preprocess_dataframe(df, drop_cols):
    df = df.drop(columns=drop_cols, errors='ignore')
    label_cols = [c for c in df.columns if 'label' in c.strip().lower()]
    if not label_cols:
        raise ValueError("No se encontró columna de etiqueta ('Label') en el DataFrame")
    label_col = label_cols[0]
    Y = df[label_col].astype(str).str.strip()
    X = df.drop(columns=[label_col])
    X = X.apply(pd.to_numeric, errors='coerce', downcast='float')
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.fillna(0)
    return X, Y

# ─── EVALUAR UN MODELO ───────────────────────────────────────────────────────
def evaluar_modelo(model, X_test, y_test, le, nombre):
    t0 = time.time()
    y_pred = model.predict(X_test)
    t1 = time.time()
    inference_time = t1 - t0

    acc  = accuracy_score(y_test, y_pred)
    f1w  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    f1m  = f1_score(y_test, y_pred, average='macro', zero_division=0)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)
    
    clases_limpias = [sanitizar_etiqueta(c) for c in le.classes_]
    cr = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)

    print(f"\n{'='*55}")
    print(f"  RESULTADOS: {nombre}")
    print(f"{'='*55}")
    print(classification_report(y_test, y_pred, target_names=clases_limpias, zero_division=0))

    return {
        'nombre': nombre,
        'y_pred': y_pred,
        'accuracy': acc,
        'f1_weighted': f1w,
        'f1_macro': f1m,
        'precision': prec,
        'recall': rec,
        'confusion_matrix': cm,
        'classification_report': cr,
        'inference_time': inference_time,
        'clases_limpias': clases_limpias
    }

# ─── FUNCIONES DE GRÁFICOS ───────────────────────────────────────────────────
def plot_confusion_matrix(ax, cm, classes, title, color_main):
    cmap = LinearSegmentedColormap.from_list('custom', ['#FFFFFF', color_main])
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            label = f'{val:,}' if val < 1e6 else f'{val/1e3:.0f}k'
            ax.text(j, i, label, ha='center', va='center', fontsize=6,
                    color='white' if val > thresh else 'black')

    ax.set_ylabel('Etiqueta Real', fontsize=9)
    ax.set_xlabel('Predicción', fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_metricas_barras(ax, resultados_rf, resultados_dt):
    metricas = ['accuracy', 'f1_weighted', 'f1_macro', 'precision', 'recall']
    # Traducido al español
    etiquetas = ['Exactitud', 'F1\n(ponderado)', 'F1\n(macro)', 'Precisión\n(ponderada)', 'Sensibilidad\n(ponderada)']
    x = np.arange(len(metricas))
    w = 0.35
    vals_rf = [resultados_rf[m] for m in metricas]
    vals_dt = [resultados_dt[m] for m in metricas]

    bars_rf = ax.bar(x - w/2, vals_rf, w, label='Random Forest', color=COLOR_RF, alpha=0.85, edgecolor='white', linewidth=0.8)
    bars_dt = ax.bar(x + w/2, vals_dt, w, label='Decision Tree', color=COLOR_DT, alpha=0.85, edgecolor='white', linewidth=0.8)

    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, fontsize=9)
    ax.set_ylabel('Valor', fontsize=10)
    ax.set_title('Comparativa de Métricas Globales', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_facecolor(COLOR_BG)
    ax.grid(axis='y', color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    for bar in bars_rf:
        ax.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=7, color=COLOR_RF, fontweight='bold')
    for bar in bars_dt:
        ax.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=7, color=COLOR_DT, fontweight='bold')


def plot_f1_por_clase(ax, cr_rf, cr_dt, classes_orig):
    classes_filtered = [c for c in classes_orig if c in cr_rf and c in cr_dt]
    classes_limpias  = [sanitizar_etiqueta(c) for c in classes_filtered]
    
    f1_rf = [cr_rf[c]['f1-score'] for c in classes_filtered]
    f1_dt = [cr_dt[c]['f1-score'] for c in classes_filtered]
    x = np.arange(len(classes_filtered))
    w = 0.35
    ax.bar(x - w/2, f1_rf, w, label='Random Forest', color=COLOR_RF, alpha=0.85, edgecolor='white')
    ax.bar(x + w/2, f1_dt, w, label='Decision Tree', color=COLOR_DT, alpha=0.85, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(classes_limpias, rotation=40, ha='right', fontsize=7)
    ax.set_ylabel('F1-score', fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_title('F1-score por Clase', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_facecolor(COLOR_BG)
    ax.grid(axis='y', color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_radar(ax, resultados_rf, resultados_dt):
    # Traducido al español
    categorias = ['Exactitud', 'F1\n(ponderado)', 'F1\n(macro)', 'Precisión', 'Sensibilidad']
    keys = ['accuracy', 'f1_weighted', 'f1_macro', 'precision', 'recall']
    N = len(categorias)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    vals_rf = [resultados_rf[k] for k in keys] + [resultados_rf[keys[0]]]
    vals_dt = [resultados_dt[k] for k in keys] + [resultados_dt[keys[0]]]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categorias, size=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=6, color='grey')
    ax.grid(color=COLOR_GRID)

    ax.plot(angles, vals_rf, 'o-', linewidth=2, color=COLOR_RF, label='Random Forest')
    ax.fill(angles, vals_rf, alpha=0.15, color=COLOR_RF)
    ax.plot(angles, vals_dt, 'o-', linewidth=2, color=COLOR_DT, label='Decision Tree')
    ax.fill(angles, vals_dt, alpha=0.15, color=COLOR_DT)
    ax.set_title('Diagrama Radar de Métricas', fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)


def plot_tiempo_inferencia(ax, resultados_rf, resultados_dt, n_test):
    modelos = ['Random Forest', 'Decision Tree']
    tiempos = [resultados_rf['inference_time'], resultados_dt['inference_time']]
    colores = [COLOR_RF, COLOR_DT]

    bars = ax.bar(modelos, tiempos, color=colores, alpha=0.85, edgecolor='white', linewidth=0.8, width=0.4)
    ax.set_ylabel('Tiempo (segundos)', fontsize=10)
    ax.set_title(f'Tiempo de Inferencia\n({n_test:,} muestras)', fontsize=11, fontweight='bold')
    ax.set_facecolor(COLOR_BG)
    ax.grid(axis='y', color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    for bar, t in zip(bars, tiempos):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{t:.3f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')


def plot_tabla_resumen(ax, resultados_rf, resultados_dt):
    ax.axis('off')
    cols = ['Métrica', 'Random Forest', 'Decision Tree', 'Diferencia (RF-DT)']
    # Traducido al español
    keys = [
        ('Exactitud',                'accuracy'),
        ('F1-score (ponderado)',     'f1_weighted'),
        ('F1-score (macro)',         'f1_macro'),
        ('Precisión (ponderada)',    'precision'),
        ('Sensibilidad (ponderada)','recall'),
    ]
    data = []
    for nombre, key in keys:
        rf_val = resultados_rf[key]
        dt_val = resultados_dt[key]
        diff   = rf_val - dt_val
        signo  = '+' if diff >= 0 else ''
        data.append([nombre, f'{rf_val:.4f}', f'{dt_val:.4f}', f'{signo}{diff:.4f}'])

    diff_t = resultados_rf['inference_time'] - resultados_dt['inference_time']
    signo  = '+' if diff_t >= 0 else ''
    data.append(['Tiempo inferencia (s)',
                 f"{resultados_rf['inference_time']:.3f}",
                 f"{resultados_dt['inference_time']:.3f}",
                 f"{signo}{diff_t:.3f}"])

    table = ax.table(cellText=data, colLabels=cols, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    for j in range(len(cols)):
        table[0, j].set_facecolor('#1E3A5F')
        table[0, j].set_text_props(color='white', fontweight='bold')

    for i, row in enumerate(data):
        diff_val = float(row[3])
        for j in range(len(cols)):
            cell = table[i+1, j]
            cell.set_facecolor('#F0F4FF' if i % 2 == 0 else 'white')
            if j == 3:
                if diff_val > 0.001:
                    cell.set_facecolor('#D1FAE5')
                    cell.set_text_props(color='#065F46', fontweight='bold')
                elif diff_val < -0.001:
                    cell.set_facecolor('#FEE2E2')
                    cell.set_text_props(color='#991B1B', fontweight='bold')

    ax.set_title('Tabla Resumen Comparativa', fontsize=12, fontweight='bold', pad=20)


# ─── GENERACIÓN DEL PDF ──────────────────────────────────────────────────────
def generar_pdf_comparativa(resultados_rf, resultados_dt, le, output_path='comparativa_modelos_TFG.pdf'):
    classes_orig = list(le.classes_)
    clases_limpias = resultados_rf['clases_limpias']
    cr_rf  = resultados_rf['classification_report']
    cr_dt  = resultados_dt['classification_report']
    n_test = len(resultados_rf['y_pred'])

    with PdfPages(output_path) as pdf:
        # ── PÁGINA 1: Portada + Tabla resumen ─────────────────────────────
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor(COLOR_BG)
        gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.4, top=0.88, bottom=0.08)

        fig.text(0.5, 0.95, 'Comparativa de Modelos de Clasificación de Tráfico de Red',
                 ha='center', va='top', fontsize=16, fontweight='bold', color='#1E3A5F')
        fig.text(0.5, 0.91, 'Random Forest  vs  Decision Tree  |  Dataset: CIC-IDS-2017',
                 ha='center', va='top', fontsize=11, color='#475569')

        ax_tabla = fig.add_subplot(gs[0])
        plot_tabla_resumen(ax_tabla, resultados_rf, resultados_dt)

        ax_barras = fig.add_subplot(gs[1])
        plot_metricas_barras(ax_barras, resultados_rf, resultados_dt)

        pdf.savefig(fig, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

        # ── PÁGINA 2: Matrices de confusión ───────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.patch.set_facecolor(COLOR_BG)
        fig.suptitle('Matrices de Confusión', fontsize=15, fontweight='bold', color='#1E3A5F', y=1.01)

        plot_confusion_matrix(axes[0], resultados_rf['confusion_matrix'], clases_limpias,
                               f"Random Forest\nExactitud: {resultados_rf['accuracy']:.4f}", COLOR_RF)
        plot_confusion_matrix(axes[1], resultados_dt['confusion_matrix'], clases_limpias,
                               f"Decision Tree\nExactitud: {resultados_dt['accuracy']:.4f}", COLOR_DT)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

        # ── PÁGINA 3: F1 por clase + Radar + Tiempo ───────────────────────
        fig = plt.figure(figsize=(16, 7))
        fig.patch.set_facecolor(COLOR_BG)
        fig.suptitle('Análisis Detallado por Clase y Rendimiento General',
                     fontsize=14, fontweight='bold', color='#1E3A5F')

        gs3 = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35,
                                left=0.06, right=0.97, top=0.88, bottom=0.18)

        ax_f1    = fig.add_subplot(gs3[0])
        plot_f1_por_clase(ax_f1, cr_rf, cr_dt, classes_orig)

        ax_radar = fig.add_subplot(gs3[1], polar=True)
        plot_radar(ax_radar, resultados_rf, resultados_dt)

        ax_tiempo = fig.add_subplot(gs3[2])
        plot_tiempo_inferencia(ax_tiempo, resultados_rf, resultados_dt, n_test)

        pdf.savefig(fig, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

        # ── PÁGINA 4: Precisión y Recall por clase ────────────────────────
        classes_filtered = [c for c in classes_orig if c in cr_rf and c in cr_dt]
        classes_limpias_2 = [sanitizar_etiqueta(c) for c in classes_filtered]
        prec_rf = [cr_rf[c]['precision'] for c in classes_filtered]
        rec_rf  = [cr_rf[c]['recall']    for c in classes_filtered]
        prec_dt = [cr_dt[c]['precision'] for c in classes_filtered]
        rec_dt  = [cr_dt[c]['recall']    for c in classes_filtered]
        x = np.arange(len(classes_filtered))
        w = 0.35

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor(COLOR_BG)
        fig.suptitle('Precisión y Sensibilidad por Clase', fontsize=14, fontweight='bold', color='#1E3A5F')

        for ax, vals_rf_, vals_dt_, metrica in [
            (axes[0], prec_rf, prec_dt, 'Precisión'),
            (axes[1], rec_rf,  rec_dt,  'Sensibilidad')
        ]:
            ax.bar(x - w/2, vals_rf_, w, label='Random Forest', color=COLOR_RF, alpha=0.85, edgecolor='white')
            ax.bar(x + w/2, vals_dt_, w, label='Decision Tree',  color=COLOR_DT, alpha=0.85, edgecolor='white')
            ax.set_xticks(x)
            ax.set_xticklabels(classes_limpias_2, rotation=40, ha='right', fontsize=7)
            ax.set_ylim(0, 1.12)
            ax.set_ylabel(metrica, fontsize=10)
            ax.set_title(f'{metrica} por Clase', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.set_facecolor(COLOR_BG)
            ax.grid(axis='y', color=COLOR_GRID, linewidth=0.8)
            ax.set_axisbelow(True)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

        # Metadatos del PDF
        d = pdf.infodict()
        d['Title']   = 'Comparativa RF vs DT - CIC-IDS-2017'
        d['Author']  = 'TFG - Detección de Intrusiones'
        d['Subject'] = 'Evaluación de modelos de clasificación de tráfico de red'

    print(f"\n✅ PDF generado correctamente: {output_path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(42)

    # 1. Cargar dataset
    print("Cargando dataset...")
    df = load_cic_dataset()
    print(f"Total de registros: {len(df):,}")
    X, Y = preprocess_dataframe(df, columnas_a_eliminar)

    # 2. Cargar label encoder
    print("Cargando label encoder...")
    le = joblib.load("label_encoder.pkl")
    y_encoded = le.transform(Y)
    
    clases_limpias = [sanitizar_etiqueta(c) for c in le.classes_]
    print("Clases limpias a renderizar:", clases_limpias)

    # 3. Mismo split que en el entrenamiento
    _, X_test, _, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Muestras de test: {len(X_test):,}")

    # 4. Cargar modelos
    print("\nCargando modelos...")
    rf = joblib.load("random_forest_cic17_best.pkl")
    dt = joblib.load("decision_tree_cic17.pkl")

    # Alinear columnas
    feat_rf = getattr(rf, "feature_names_in_", None)
    feat_dt = getattr(dt, "feature_names_in_", None)

    X_test_rf = X_test.reindex(columns=feat_rf, fill_value=0) if feat_rf is not None else X_test
    X_test_dt = X_test.reindex(columns=feat_dt, fill_value=0) if feat_dt is not None else X_test

    print(f"  Random Forest: {X_test_rf.shape[1]} features")
    print(f"  Decision Tree: {X_test_dt.shape[1]} features")

    # 5. Evaluar
    print("\nEvaluando modelos...")
    res_rf = evaluar_modelo(rf, X_test_rf, y_test, le, "Random Forest")
    res_dt = evaluar_modelo(dt, X_test_dt, y_test, le, "Decision Tree")

    # 6. Generar PDF
    print("\nGenerando PDF comparativo para el TFG...")
    generar_pdf_comparativa(res_rf, res_dt, le, output_path='comparativa_modelos_TFG.pdf')

if __name__ == "__main__":
    main()