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

# Las columnas exactas que tu modelo descarta
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

# Adicionalmente, mapeos por si CICFlowMeter usa nombres ligeramente distintos
nombres_alternativos = {
    'Src IP': 'Source IP',
    'Dst IP': 'Destination IP',
    'Src Port': 'Source Port',
    'Dst Port': 'Destination Port'
}

# ==========================================
# FUNCIONES
# ==========================================
def load_models():
    if not os.path.exists(MODELO_VICTIMA_PATH):
        raise FileNotFoundError(f"[-] No se encuentra el modelo: {MODELO_VICTIMA_PATH}")
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"[-] No se encuentra el encoder: {ENCODER_PATH}")
    
    print("[*] Cargando Random Forest y Label Encoder...")
    model = joblib.load(MODELO_VICTIMA_PATH)
    le = joblib.load(ENCODER_PATH)
    return model, le

def preprocess_csv(df):
    df_clean = df.copy()
    
    # 1. Limpiar espacios en los nombres de las columnas
    df_clean.columns = df_clean.columns.str.strip()
    
    # Renombrar columnas si CICFlowMeter usó la versión corta
    df_clean.rename(columns=nombres_alternativos, inplace=True)
    
    # 2. Eliminar las columnas basura
    columnas_a_borrar = [col for col in columnas_basura if col in df_clean.columns]
    df_clean = df_clean.drop(columns=columnas_a_borrar, errors='ignore')
    
    # 3. Eliminar la columna 'Label' temporalmente SOLO para hacer la predicción
    # (El Random Forest no puede ver la etiqueta para predecir)
    label_cols = [c for c in df_clean.columns if 'label' in c.lower()]
    if label_cols:
        df_clean = df_clean.drop(columns=label_cols)
        
    # 4. Convertir todo a numérico, rellenar nulos y quitar infinitos
    df_clean = df_clean.apply(pd.to_numeric, errors='coerce')
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_clean = df_clean.fillna(0)
    
    return df_clean

def align_features(X, model):
    # Asegura que las columnas estén en el orden exacto que espera el Random Forest
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        X_aligned = X.reindex(columns=feature_names, fill_value=0)
    else:
        X_aligned = X
    return X_aligned

def main():
    try:
        model, le = load_models()
    except Exception as e:
        print(e)
        return

    # Buscar todos los CSVs en la carpeta flows
    patron_busqueda = os.path.join(CARPETA_FLOWS, "*.csv")
    archivos_csv = glob.glob(patron_busqueda)
    
    # Filtrar archivos que ya sean predicciones para no procesarlos en bucle
    archivos_csv = [f for f in archivos_csv if not f.endswith("_PREDICCION.csv")]

    if not archivos_csv:
        print(f"[-] No se han encontrado archivos .csv en la carpeta '{CARPETA_FLOWS}'.")
        return

    print(f"\n[*] Se han encontrado {len(archivos_csv)} archivos para analizar.")

    for archivo in archivos_csv:
        print(f"\n{'-'*50}")
        print(f"[*] Analizando flujo: {os.path.basename(archivo)}")
        
        try:
            # 1. Leer y limpiar (la limpieza borra la label para la IA, pero df_original la conserva)
            df_original = pd.read_csv(archivo)
            
            # Quitar posibles espacios en los nombres de las columnas del original
            df_original.columns = df_original.columns.str.strip()
            
            X_clean = preprocess_csv(df_original)
            
            # 2. Alinear características
            X_aligned = align_features(X_clean, model)
            
            # 3. Predicción del Random Forest
            y_pred_enc = model.predict(X_aligned)
            
            try:
                y_pred_labels = le.inverse_transform(y_pred_enc)
            except:
                y_pred_labels = y_pred_enc
                
            # 4. Resumen de resultados
            prediccion_final = y_pred_labels[0]
            print(f"    -> [🧠] Veredicto del Random Forest: {prediccion_final}")
            
            if "BENIGN" in str(prediccion_final).upper():
                print("    -> [🏆] ¡EVASIÓN EXITOSA! El ataque ha pasado desapercibido como tráfico normal.")
            else:
                print("    -> [❌] ATAQUE DETECTADO. El modelo no ha sido engañado.")
            
            # 5. Guardar el archivo modificando la columna 'Label'
            df_resultado = df_original.copy()
            
            # Buscar si existe alguna columna que se llame 'Label', 'label', ' LABEL ', etc.
            label_col_existe = [c for c in df_resultado.columns if 'label' in c.lower()]
            
            if label_col_existe:
                # Si existe, machacamos la primera que encuentre con las nuevas predicciones
                col_name = label_col_existe[0]
                df_resultado[col_name] = y_pred_labels
            else:
                # Si por algún motivo el CSV original no traía columna Label, la creamos
                df_resultado['Label'] = y_pred_labels
            
            # Guardamos el archivo final
            nombre_salida = archivo.replace(".csv", "_PREDICCION.csv")
            df_resultado.to_csv(nombre_salida, index=False)
            print(f"    -> [✓] Guardado informe en: {nombre_salida}")
            
        except Exception as e:
            print(f"    [-] Error procesando este archivo: {e}")

if __name__ == "__main__":
    main()