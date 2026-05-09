import os
import glob
import joblib
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURACIÓN
# ==========================================
CARPETA_FLOWS = 'flows'
MODELO_VICTIMA_PATH = 'random_forest_cic17_best.pkl'
ENCODER_PATH = 'label_encoder.pkl'
SCALER_PATH = 'scaler.pkl'  # <-- IMPORTANTE: Cargar el escalador original si se usa
ARCHIVO_REPORTE_FINAL = 'REPORTE_GLOBAL_PREDICCIONES.csv'

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

nombres_alternativos = {
    'Src IP': 'Source IP',
    'Dst IP': 'Destination IP',
    'Src Port': 'Source Port',
    'Dst Port': 'Destination Port'
}

def load_models():
    if not os.path.exists(MODELO_VICTIMA_PATH):
        raise FileNotFoundError(f"[-] No se encuentra el modelo: {MODELO_VICTIMA_PATH}")
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"[-] No se encuentra el encoder: {ENCODER_PATH}")
    if not os.path.exists(SCALER_PATH):
        print("[!] Advertencia: No se encontró scaler.pkl. Las predicciones podrían fallar.")
        scaler = None
    else:
        scaler = joblib.load(SCALER_PATH)
    
    print("[*] Cargando Random Forest, Label Encoder y Scaler...")
    model = joblib.load(MODELO_VICTIMA_PATH)
    le = joblib.load(ENCODER_PATH)
    return model, le, scaler

def preprocess_csv(df):
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.strip()
    df_clean.rename(columns=nombres_alternativos, inplace=True)
    
    columnas_a_borrar = [col for col in columnas_basura if col in df_clean.columns]
    df_clean = df_clean.drop(columns=columnas_a_borrar, errors='ignore')
    
    label_cols = [c for c in df_clean.columns if 'label' in c.lower()]
    if label_cols:
        df_clean = df_clean.drop(columns=label_cols)
        
    df_clean = df_clean.apply(pd.to_numeric, errors='coerce')
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_clean = df_clean.fillna(0)
    
    return df_clean

def align_features(X, model):
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        X_aligned = X.reindex(columns=feature_names, fill_value=0)
    else:
        X_aligned = X
    return X_aligned

def main():
    try:
        model, le, scaler = load_models()
    except Exception as e:
        print(e)
        return

    patron_busqueda = os.path.join(CARPETA_FLOWS, "*.csv")
    archivos_csv = glob.glob(patron_busqueda)
    
    if not archivos_csv:
        print(f"[-] No se han encontrado archivos .csv en la carpeta '{CARPETA_FLOWS}'.")
        return

    print(f"\n[*] Se han encontrado {len(archivos_csv)} archivos para analizar.")
    
    lista_resultados = []
    evasiones_exitosas = 0
    total_analizados = 0

    for archivo in archivos_csv:
        nombre_base = os.path.basename(archivo)
        
        try:
            df_original = pd.read_csv(archivo)
            
            if df_original.empty:
                continue
                
            df_original.columns = df_original.columns.str.strip()

            # --- CAMBIO 1: FILTRAR CABECERAS REPETIDAS ---
            # Si CICFlowMeter concatenó archivos, eliminamos las filas con nombres de columnas
            if 'Flow ID' in df_original.columns:
                df_original = df_original[df_original['Flow ID'] != 'Flow ID'].copy()

            if df_original.empty:
                continue
            
            X_clean = preprocess_csv(df_original)
            X_aligned = align_features(X_clean, model)
            
            # --- CAMBIO 2: ESCALAR ANTES DE PREDECIR ---
            if scaler:
                X_final = scaler.transform(X_aligned)
            else:
                X_final = X_aligned

            y_pred_enc = model.predict(X_final)
            
            try:
                y_pred_labels = le.inverse_transform(y_pred_enc)
            except:
                y_pred_labels = y_pred_enc
                
            df_resultado = df_original.copy()
            
            # --- ASIGNACIÓN DE VERDICTO ---
            df_resultado['Veredicto_IA'] = y_pred_labels
            df_resultado.insert(0, 'Archivo_Origen', nombre_base)
            
            # Conteo de evasiones (si hay múltiples filas por archivo, contamos la tasa)
            flujos_en_archivo = len(y_pred_labels)
            evasiones_en_archivo = np.sum(["BENIGN" in str(lb).upper() for lb in y_pred_labels])
            
            total_analizados += flujos_en_archivo
            evasiones_exitosas += evasiones_en_archivo

            if evasiones_en_archivo == flujos_en_archivo:
                print(f"[🏆] {nombre_base}: EVASIÓN TOTAL ({evasiones_en_archivo}/{flujos_en_archivo})")
            else:
                print(f"[❌] {nombre_base}: DETECTADO ({flujos_en_archivo - evasiones_en_archivo} flujos detectados)")
                
            lista_resultados.append(df_resultado)
            
        except Exception as e:
            print(f"[-] {nombre_base}: Error procesando archivo: {e}")

    if lista_resultados:
        print(f"\n{'='*50}")
        print("[*] Generando archivo de reporte global único...")
        
        df_final_global = pd.concat(lista_resultados, ignore_index=True)
        df_final_global.to_csv(ARCHIVO_REPORTE_FINAL, index=False)
        
        print(f"[✓] Reporte guardado exitosamente en: {ARCHIVO_REPORTE_FINAL}")
        print(f"[i] TASA DE EVASIÓN GLOBAL: {(evasiones_exitosas/total_analizados)*100:.2f}% ({evasiones_exitosas}/{total_analizados} flujos)")
    else:
        print("\n[-] No se pudo procesar ningún archivo con éxito.")

if __name__ == "__main__":
    main()