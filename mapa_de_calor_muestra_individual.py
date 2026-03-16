import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import random

# ==========================================
# 1. CONFIGURACIÓN Y FILTROS
# ==========================================
ARCHIVO_MODIFICADO = "ataques_adversarios_COMPLETOS.csv"
ARCHIVO_REPORTE = "REPORTE_GAN.csv"
DIRECTORIO_REAL = "Dataset/MachineLearningCVE"

# Columnas basura que no existen en las generadas o son irrelevantes
columnas_basura = [
    'Destination Port', 'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Timestamp', 
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    'URG Flag Count', 'CWE Flag Count', 'CWR Flag Count', 'ECE Flag Count',
    'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Fwd Avg Bytes/Bulk',
    'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
    'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
    'Fwd Seg Size Min', 'min_seg_size_forward', 'Fwd Header Length.1'
]

def generar_comparativa_aleatoria_total():
    # 1. Cargar archivo de ataques (FUENTE DE VERDAD)
    if not os.path.exists(ARCHIVO_MODIFICADO):
        print(f"[-] No se encontró el archivo: {ARCHIVO_MODIFICADO}")
        return
    
    df_ataques = pd.read_csv(ARCHIVO_MODIFICADO)
    
    # --- SELECCIÓN ALEATORIA DE MUESTRA ---
    # Usamos el índice de este dataframe para alinear todo
    idx_row = random.randint(0, len(df_ataques) - 1)
    
    # Extraemos los metadatos directamente del archivo de ataques
    try:
        target_file = df_ataques.iloc[idx_row]['_ORIGIN_FILE_']
        target_idx = int(df_ataques.iloc[idx_row]['_ORIGIN_IDX_'])
        m_mod = df_ataques.iloc[[idx_row]] # Mantenemos como DataFrame
    except KeyError:
        print("[-] Error: El archivo de ataques no tiene las columnas '_ORIGIN_FILE_' o '_ORIGIN_IDX_'.")
        print("    Asegúrate de estar usando el CSV generado por el script de la GAN corregido.")
        return

    print(f"[*] Seleccionada muestra aleatoria (Fila {target_idx} de {target_file})")

    # 2. Cargar muestra original del dataset REAL
    ruta_orig = os.path.join(DIRECTORIO_REAL, target_file)
    m_orig = pd.DataFrame()

    if os.path.exists(ruta_orig):
        try:
            df_orig = pd.read_csv(ruta_orig)
            df_orig.columns = [c.strip() for c in df_orig.columns]
            m_orig = df_orig.iloc[[target_idx]]
        except Exception as e:
            print(f"[!] Error leyendo original: {e}")
    
    # 3. Fallback: Si no carga el original, usar REPORTE_GAN.csv
    if m_orig.empty:
        print(f"[!] Aviso: Dataset original no accesible. Intentando recuperar datos desde {ARCHIVO_REPORTE}...")
        if os.path.exists(ARCHIVO_REPORTE):
            reporte = pd.read_csv(ARCHIVO_REPORTE)
            # Asumimos que el orden de filas es el mismo (1 a 1)
            info = reporte.iloc[idx_row] 
            
            cols_old = [c for c in reporte.columns if c.startswith('OLD_')]
            features_base = [c.replace('OLD_', '') for c in cols_old]
            
            # Reconstruir m_orig desde el reporte
            m_orig = pd.DataFrame([info[cols_old].values], columns=features_base)
            
            # Filtrar m_mod para que coincida
            m_mod = m_mod[[c for c in m_mod.columns if c in features_base]]
        else:
            print("[-] No se puede reconstruir la muestra original.")
            return

    # 4. Limpieza y Alineación de Columnas
    ignorar = columnas_basura + ['Label', '_ORIGIN_FILE_', '_ORIGIN_IDX_']
    cols = [c for c in m_mod.columns if c in m_orig.columns and c not in ignorar]
    
    # 5. Procesamiento de Datos (Real + Normalizado)
    datos_completos = []
    for c in cols:
        try:
            v_orig = float(m_orig[c].values[0])
            v_mod = float(m_mod[c].values[0])
            
            # Evitar división por cero
            limit = max(abs(v_orig), abs(v_mod))
            if limit == 0: limit = 1.0
            
            datos_completos.append({
                'Característica': c,
                'Valor_Original': v_orig,
                'Valor_Sintetico': v_mod,
                'Normalizado_Original': v_orig / limit,
                'Normalizado_Sintetico': v_mod / limit,
                'Diferencia_Absoluta': abs(v_orig - v_mod),
                'Diferencia_Normalizada': abs((v_orig - v_mod) / limit)
            })
        except Exception as e:
            continue

    df_resultado = pd.DataFrame(datos_completos)
    
    if df_resultado.empty:
        print("[-] No hay datos para graficar.")
        return

    # --- EXPORTAR CSV ---
    nombre_csv = f"datos_comparativos_fila_{target_idx}.csv"
    df_resultado.to_csv(nombre_csv, index=False)
    print(f"[✓] CSV guardado: {nombre_csv}")

    # 6. GENERAR GRÁFICO (Top 40 Cambios)
    df_plot = df_resultado.sort_values(by='Diferencia_Normalizada', ascending=False).head(40)
    
    df_melt = df_plot.melt(id_vars='Característica', 
                           value_vars=['Normalizado_Original', 'Normalizado_Sintetico'],
                           var_name='Versión', value_name='Valor (0-1)')
    
    df_melt['Versión'] = df_melt['Versión'].replace({'Normalizado_Original': 'Original', 
                                                     'Normalizado_Sintetico': 'Sintética'})

    plt.figure(figsize=(12, 11))
    sns.barplot(data=df_melt, y='Característica', x='Valor (0-1)', hue='Versión', palette='coolwarm')
    plt.title(f"Comparativa Normalizada: Fila {target_idx}\n{target_file}", fontsize=14)
    plt.xlim(0, 1.1)
    plt.grid(True, axis='x', ls="--", alpha=0.5)
    plt.tight_layout()
    
    nombre_png = f"comparativa_fila_{target_idx}.png"
    plt.savefig(nombre_png, dpi=300)
    print(f"[✓] Gráfico guardado: {nombre_png}")

if __name__ == "__main__":
    generar_comparativa_aleatoria_total()