import os
from glob import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import time

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

def load_cic_dataset(path_pattern="Dataset/MachineLearningCVE/*.csv"):
    files = glob(path_pattern)
    if not files:
        raise FileNotFoundError(f"No CSV files found for pattern: {path_pattern}")
    df_list = [pd.read_csv(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    df.columns = df.columns.str.strip()
    return df

def preprocess_dataframe(df, drop_cols):
    # Eliminamos las columnas indicadas
    df = df.drop(columns=drop_cols, errors='ignore')
    
    # detect label column (handles 'Label' or ' Label')
    label_cols = [c for c in df.columns if 'label' in c.strip().lower()]
    if not label_cols:
        raise ValueError("No se encontró columna de etiqueta ('Label') en el DataFrame")
    label_col = label_cols[0]
    
    Y = df[label_col].astype(str).str.strip()
    X = df.drop(columns=[label_col])
    
    # Conversión a numérico optimizada (float32 para ahorrar RAM)
    X = X.apply(pd.to_numeric, errors='coerce', downcast='float')
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.fillna(0)
    return X, Y

def main():
    np.random.seed(42)

    # Cargar y preprocesar
    print("Cargando dataset...")
    df = load_cic_dataset()
    print("Total de registros:", len(df))
    X, Y = preprocess_dataframe(df, columnas_a_eliminar)

    # Codificar etiquetas
    le = LabelEncoder()
    y_encoded = le.fit_transform(Y)
    print("Clases detectadas:", list(le.classes_))

    # División estratificada (20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # --- Ajuste de hiperparámetros rápido para datasets grandes ---
    # Reducimos muestra para que el GridSearch no tarde horas
    max_tune_samples = 20000 
    tune_n = min(max_tune_samples, X_train.shape[0])

    if tune_n < X_train.shape[0]:
        X_tune, _, y_tune, _ = train_test_split(
            X_train, y_train, train_size=tune_n, stratify=y_train, random_state=42
        )
        print(f"Usando muestra estratificada de {tune_n} muestras para búsqueda de hiperparámetros")
    else:
        X_tune, y_tune = X_train, y_train
        print("Usando todo X_train para búsqueda de hiperparámetros")

    # Parámetros específicos para Random Forest
    param_dist = {
        'n_estimators': [50, 100],      # Reducido para tuning rápido
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True]
    }

    base_rf = RandomForestClassifier(class_weight='balanced', random_state=42)

    # Configuración de RandomizedSearchCV (Optimizado para velocidad)
    n_iter_search = 10 
    rnd_search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_dist,
        n_iter=n_iter_search,
        scoring='f1_weighted',
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    print("Iniciando búsqueda de hiperparámetros...")
    t0 = time.time()
    rnd_search.fit(X_tune, y_tune)
    t1 = time.time()
    print(f"RandomizedSearchCV completado en {t1 - t0:.1f} s")
    print("Mejores hiperparámetros (de la muestra):", rnd_search.best_params_)

    # Reentrenar modelo final sobre todo X_train
    best_params = rnd_search.best_params_
    # Forzamos un mínimo de árboles para el modelo final aunque el tuning haya dicho menos
    if best_params['n_estimators'] < 100:
        best_params['n_estimators'] = 100
        
    final_rf = RandomForestClassifier(**best_params, class_weight='balanced', random_state=42, n_jobs=-1)
    
    print("Entrenando modelo final Random Forest sobre todo el conjunto de entrenamiento...")
    t0 = time.time()
    final_rf.fit(X_train, y_train)
    t1 = time.time()
    print(f"Modelo final entrenado en {t1 - t0:.1f} s")

    # Evaluación sobre test
    y_pred = final_rf.predict(X_test)
    print("Score en test (accuracy):", final_rf.score(X_test, y_test))
    print(classification_report(y_test, y_pred, target_names=list(le.classes_)))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de confusión (Random Forest)')
    plt.tight_layout()
    plt.show()

    # Guardar modelo con nombre actualizado
    model_filename = "random_forest_cic17_best.pkl"
    joblib.dump(final_rf, model_filename)
    joblib.dump(le, "label_encoder.pkl")
    print(f"Modelo guardado como '{model_filename}' y codificador actualizado.")

    # Aplicar modelo a otro CSV de tráfico (opcional)
    traffic_path = "Dataset/TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    if os.path.exists(traffic_path):
        print(f"Probando modelo en archivo externo: {traffic_path}")
        traffic = pd.read_csv(traffic_path)
        traffic.columns = traffic.columns.str.strip()
        traffic = traffic.drop(columns=columnas_a_eliminar, errors='ignore')
        
        traffic_label_cols = [c for c in traffic.columns if 'label' in c.strip().lower()]
        if traffic_label_cols:
            X_new = traffic.drop(columns=[traffic_label_cols[0]])
        else:
            X_new = traffic
            
        X_new = X_new.apply(pd.to_numeric, errors='coerce', downcast='float')
        X_new.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_new = X_new.fillna(0)
        
        feature_names = getattr(final_rf, "feature_names_in_", None)
        if feature_names is not None:
            # Reindexar para asegurar que las columnas coinciden con el entrenamiento
            X_new = X_new.reindex(columns=feature_names, fill_value=0)
            
        y_pred_new = final_rf.predict(X_new)
        y_pred_labels = le.inverse_transform(y_pred_new)
        print("Predicciones sobre tráfico externo (conteo):")
        print(pd.Series(y_pred_labels).value_counts())
    else:
        print("Archivo de tráfico externo no encontrado, omitiendo esa predicción.")

if __name__ == "__main__":
    main()