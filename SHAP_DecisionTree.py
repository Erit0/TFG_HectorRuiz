"""
SHAP Analysis para Random Forest - LISTADO COMPLETO CON PORCENTAJES
Identifica todas las características y su peso porcentual para la clase BENIGN
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURACIÓN
# ==========================================

# --- LISTA DEFINITIVA DE COLUMNAS A ELIMINAR ---
# Basada en análisis de Victor Matesanz, Miguel García y problemas conocidos de CICFlowMeter
columnas_a_eliminar = [
    # --- 1. Identificadores (Evitar Sesgo de Origen/Destino) ---
    # Miguel y Víctor eliminan esto para que el modelo no memorice IPs o puertos específicos.
    'Destination Port', 
    'Flow ID', 
    'Source IP', 
    'Source Port', 
    'Destination IP', 
    'Timestamp',  # CRÍTICO: Evita que aprenda fechas.

    # --- 2. Flags Constantes o Irrelevantes (Varianza ~0) ---
    # Estas columnas suelen valer siempre 0 en CICFlowMeter.
    'Fwd PSH Flags', 
    'Bwd PSH Flags', 
    'Fwd URG Flags', 
    'Bwd URG Flags',
    'URG Flag Count', 
    'CWE Flag Count', 
    'CWR Flag Count', # A veces aparece con este nombre
    'ECE Flag Count',

    # --- 3. Características "Bulk" (Ráfaga) ---
    # Miguel confirma que estas métricas suelen estar rotas (0) en la herramienta.
    'Fwd Avg Packets/Bulk', 
    'Fwd Avg Bulk Rate', 
    'Fwd Avg Bytes/Bulk',
    'Bwd Avg Bytes/Bulk', 
    'Bwd Avg Packets/Bulk', 
    'Bwd Avg Bulk Rate',

    # --- 4. Subflujos Redundantes ---
    # Eliminar TODOS los subflujos porque son copias exactas de los totales.
    'Subflow Fwd Packets',
    'Subflow Fwd Bytes',
    'Subflow Bwd Packets',
    'Subflow Bwd Bytes',

    # --- 5. Tiempos Active / Idle (Evitar Sesgo de Timestamp) ---
    # Víctor descubrió que 'Idle' a veces contiene fechas (1.3e15) en vez de duraciones.
    # Se eliminan TODAS para evitar contaminación.
    'Active Mean', 
    'Active Std', 
    'Active Max', 
    'Active Min',
    'Idle Mean', 
    'Idle Std', 
    'Idle Max', 
    'Idle Min',

    # --- 6. Sesgos de Sistema Operativo (Opcional pero recomendado por Víctor) ---
    # El tamaño mínimo de segmento delata si es Linux o Windows.
    'Fwd Seg Size Min', 
    'min_seg_size_forward', # Nombre alternativo común

    # --- 7. Columna Repetida (Error del CSV original) ---
    # Aunque en el archivo se llama "Fwd Header Length", Pandas la renombra
    # automáticamente a .1 al cargarla. La borramos aquí.
    'Fwd Header Length.1'
]

# Para el gráfico mantendremos un límite visual, pero el CSV y la consola mostrarán TODO.
TOP_FEATURES_PLOT = 20  
MODELO_VICTIMA_PATH = 'random_forest_cic17_best.pkl'
ENCODER_PATH = 'label_encoder.pkl'

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================

def load_cic_dataset(path_pattern="Dataset/MachineLearningCVE/*.csv"):
    print("[INFO] Cargando dataset...")
    files = glob(path_pattern)
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos CSV para: {path_pattern}")
    df_list = [pd.read_csv(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    df.columns = df.columns.str.strip()
    return df

def preprocess_dataframe(df, drop_cols):
    print("[INFO] Preprocesando datos...")
    df = df.drop(columns=drop_cols, errors='ignore')
    
    label_cols = [c for c in df.columns if 'label' in c.strip().lower()]
    if not label_cols:
        raise ValueError("No se encontró columna de etiqueta ('Label')")
    
    label_col = label_cols[0]
    Y = df[label_col].astype(str).str.strip()
    X = df.drop(columns=[label_col])
    
    X = X.apply(pd.to_numeric, errors='coerce')
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.fillna(0)
    
    return X, Y

def main():
    np.random.seed(42)
    
    print("="*60)
    print("ANÁLISIS SHAP COMPLETO: PORCENTAJE DE IMPORTANCIA")
    print("="*60)
    
    try:
        # 1. CARGAR RECURSOS
        print(f"\n[INFO] Cargando encoder y modelo...")
        if not os.path.exists(ENCODER_PATH) or not os.path.exists(MODELO_VICTIMA_PATH):
            raise FileNotFoundError("Faltan archivos del modelo (.pkl)")
            
        label_encoder = joblib.load(ENCODER_PATH)
        model = joblib.load(MODELO_VICTIMA_PATH)
        print(f"[✓] Modelo Random Forest cargado correctamente")

        # 2. IDENTIFICAR ÍNDICE DE 'BENIGN'
        try:
            posibles_nombres = ['BENIGN', 'Benign', 'benign']
            benign_label = next(l for l in label_encoder.classes_ if l in posibles_nombres)
            benign_idx = list(label_encoder.classes_).index(benign_label)
            print(f"[INFO] Clase objetivo identificada: '{benign_label}' (Índice: {benign_idx})")
        except StopIteration:
            print("[ERROR] No se encontró la clase 'BENIGN' en el encoder.")
            return

        # 3. CARGAR DATOS (Muestra)
        print(f"\n[INFO] Cargando muestra del dataset para análisis...")
        df = load_cic_dataset()
        X, y = preprocess_dataframe(df, columnas_a_eliminar)
        
        y_encoded = label_encoder.transform(y)
        
        # Muestra reducida para velocidad
        SAMPLE_SIZE = 2000
        if len(X) > SAMPLE_SIZE:
            print(f"[INFO] Seleccionando subconjunto de {SAMPLE_SIZE} muestras para SHAP...")
            X_shap, _, y_shap, _ = train_test_split(
                X, y_encoded, train_size=SAMPLE_SIZE, random_state=42, stratify=y_encoded
            )
        else:
            X_shap = X

        # 4. CÁLCULO SHAP
        print(f"\n[INFO] Iniciando cálculo de valores SHAP...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap, check_additivity=False)
        
        # 5. SELECCIÓN DE VALORES PARA CLASE BENIGNA
        print(f"[INFO] Procesando estructura de datos SHAP...")
        
        shap_values_benign = None
        
        # Manejo robusto de dimensiones (Igual que tu código original)
        if isinstance(shap_values, list):
            shap_values_benign = shap_values[benign_idx]
        elif isinstance(shap_values, np.ndarray):
            if len(shap_values.shape) == 3:
                shap_values_benign = shap_values[:, :, benign_idx]
            elif len(shap_values.shape) == 2:
                if len(label_encoder.classes_) == 2 and benign_idx == 0:
                    shap_values_benign = -shap_values 
                else:
                    shap_values_benign = shap_values
            else:
                raise ValueError(f"Formato SHAP desconocido: {shap_values.shape}")
        else:
            raise TypeError(f"Tipo SHAP no soportado: {type(shap_values)}")

        # 6. CALCULAR IMPORTANCIA Y PORCENTAJES
        # Media del valor absoluto para cada característica
        mean_abs_shap = np.abs(shap_values_benign).mean(axis=0)
        
        if mean_abs_shap.ndim > 1:
            mean_abs_shap = mean_abs_shap.flatten()

        # Crear DataFrame con TODAS las características
        feature_importance_df = pd.DataFrame({
            'Feature': list(X.columns),
            'Mean_Abs_SHAP': mean_abs_shap
        })

        # --- CÁLCULO DE PORCENTAJE ---
        total_importance = feature_importance_df['Mean_Abs_SHAP'].sum()
        feature_importance_df['Percentage'] = (feature_importance_df['Mean_Abs_SHAP'] / total_importance) * 100

        # Ordenar descendente
        feature_importance_df = feature_importance_df.sort_values('Percentage', ascending=False)

        # 7. RESULTADOS (IMPRIMIR TODOS)
        print(f"\n{'='*85}")
        print(f"LISTADO COMPLETO DE IMPORTANCIA DE CARACTERÍSTICAS (Clase: {benign_label})")
        print(f"{'='*85}")
        print(f"{'Rank':<5} | {'Característica':<40} | {'Porcentaje %':<15} | {'Impacto Bruto'}")
        print(f"{'-'*85}")
        
        for i, (idx, row) in enumerate(feature_importance_df.iterrows()):
            print(f"{i+1:<5} | {row['Feature']:<40} | {row['Percentage']:6.4f}%        | {row['Mean_Abs_SHAP']:.6f}")
            
        print(f"{'-'*85}")
        print(f"Total acumulado: {feature_importance_df['Percentage'].sum():.2f}%")

        # 8. GRAFICAR (Top N para legibilidad)
        print(f"\n[INFO] Generando gráficos (Top {TOP_FEATURES_PLOT})...")
        top_plot = feature_importance_df.head(TOP_FEATURES_PLOT)
        
        # Gráfico Beeswarm
        plt.figure()
        shap.summary_plot(shap_values_benign, X_shap, max_display=TOP_FEATURES_PLOT, show=False)
        plt.title(f'Top {TOP_FEATURES_PLOT} Características - Clase {benign_label}', fontsize=12)
        plt.tight_layout()
        plt.savefig('SHAP_Benign_Beeswarm.png', dpi=300, bbox_inches='tight')
        
        # Gráfico de barras con Porcentajes
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Percentage', y='Feature', data=top_plot, palette='viridis')
        plt.title(f'Importancia Relativa (%) - Clase {benign_label} (Top {TOP_FEATURES_PLOT})')
        plt.xlabel('Importancia (%)')
        plt.tight_layout()
        plt.savefig('SHAP_Benign_Percentage_Bars.png', dpi=300)

        # 9. GUARDAR CSV COMPLETO
        csv_filename = 'Todas_Features_SHAP_Importancia.csv'
        feature_importance_df.to_csv(csv_filename, index=False)
        print(f"\n[✓] Archivo guardado con TODAS las características: '{csv_filename}'")

    except Exception as e:
        print(f"\n[ERROR CRÍTICO] {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()