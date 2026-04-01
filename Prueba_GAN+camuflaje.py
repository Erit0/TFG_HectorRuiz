# ==========================================
# GAN ADVERSARIA: VERSIÓN LIBRE PERO DETERMINISTA (RESULTADOS FIJOS)
# ==========================================

import os
import random # <--- IMPORTANTE: Necesario para controlar el azar
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, backend as K
from sklearn.preprocessing import MinMaxScaler
import joblib


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0=Todo, 1=No Info, 2=No Warnings, 3=No Errors
# --- BLOQUE DE REPRODUCIBILIDAD (CONGELAR EL AZAR) ---
# Esto garantiza que siempre coja los mismos ataques y la GAN aprenda igual.
SEMILLA = 42

os.environ['PYTHONHASHSEED'] = str(SEMILLA)
random.seed(SEMILLA)
np.random.seed(SEMILLA)
tf.random.set_seed(SEMILLA)

# Fuerza operaciones deterministas en TensorFlow
tf.config.experimental.enable_op_determinism()

print(f"[*] Modo Determinista ACTIVADO. Semilla: {SEMILLA}")
# -----------------------------------------------------

# --- CONFIGURACIÓN ---
DATOS_REALES_DIR = 'Dataset/MachineLearningCVE'
MODELO_VICTIMA_PATH = 'random_forest_cic17_best.pkl'
ENCODER_PATH = 'label_encoder.pkl'

CANTIDAD_ATAQUES = 1000   
EPOCHS = 300            
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
FACTOR_PENALIZACION = 25.0 

# Límites físicos (Deltas máximos)
LIMITES_FISICOS = {
    'Init_Win_bytes_forward': 65535.0,
    'Init_Win_bytes_backward': 0.0, 
    'Bwd Packet Length Min': 2.0,
    'Fwd Header Length': 400.0, # Margen amplio para evitar colapso
    'Bwd Packet Length Std': 10.0,
    'Bwd Packet Length Mean': 5.0,
    'Avg Bwd Segment Size': 5.0,
    'Bwd Packet Length Max': 10.0,
    'Packet Length Mean': 800.0, #Posible para modificar 
    'Max Packet Length': 10.0,
    'Total Length of Bwd Packets': 50.0, 
    'Flow IAT Mean': 1000.0  # modificlable en mucha menor medida
}


SHAP_FEATURES = list(LIMITES_FISICOS.keys())

# Columnas basura a eliminar
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

# ==========================================
# 1. CARGA DE DATOS
# ==========================================
def cargar_datos_completos():
    print("[*] Cargando dataset completo...")
    files = glob.glob(os.path.join(DATOS_REALES_DIR, "*.csv"))
    df_list = []
    for f in files:
        try:
            df_t = pd.read_csv(f)
            df_t.columns = df_t.columns.str.strip()
            df_t['_ORIGIN_FILE_'] = os.path.basename(f)
            df_t['_ORIGIN_IDX_'] = df_t.index 
            df_list.append(df_t)
        except: pass
    
    df_full = pd.concat(df_list, ignore_index=True)
    label_col = [c for c in df_full.columns if 'label' in c.lower()][0]
    
    # Benignos
    df_benign_train = df_full[df_full[label_col] == 'BENIGN'].copy()
    for col in SHAP_FEATURES:
        if col in df_benign_train.columns:
            df_benign_train = df_benign_train[df_benign_train[col] >= 0]
    df_benign_shap = df_benign_train[SHAP_FEATURES].dropna()
    
    # Ataques
    df_attacks_full = df_full[df_full[label_col] != 'BENIGN'].dropna()
    if len(df_attacks_full) > CANTIDAD_ATAQUES:
        # Nota: random_state=42 asegura que SIEMPRE se elijan las mismas filas
        # aunque sea un muestreo aleatorio (variado).
        df_attacks_full = df_attacks_full.sample(n=CANTIDAD_ATAQUES, random_state=SEMILLA)
    
    df_attacks_shap = df_attacks_full[SHAP_FEATURES].copy()
    
    print(f"[✓] Benignos: {len(df_benign_shap)} | Ataques: {len(df_attacks_full)}")
    return df_benign_shap, df_attacks_shap, df_attacks_full

# ==========================================
# 2. GAN (Cerebro)
# ==========================================
class HierarchicalLogicLayer(layers.Layer):
    def __init__(self, idx_min, idx_mean, idx_max, **kwargs):
        super(HierarchicalLogicLayer, self).__init__(**kwargs)
        self.idx_min, self.idx_mean, self.idx_max = idx_min, idx_mean, idx_max

    def call(self, inputs):
        original, modified = inputs
        out = tf.maximum(modified, 0.0)
        out = tf.where(original > 0, tf.maximum(out, 0.001), out)
        
        features_list = tf.split(out, num_or_size_splits=len(SHAP_FEATURES), axis=1)
        
        if self.idx_mean is not None:
            mean_val = features_list[self.idx_mean]
            if self.idx_min is not None:
                features_list[self.idx_min] = tf.minimum(features_list[self.idx_min], mean_val)
            if self.idx_max is not None:
                features_list[self.idx_max] = tf.maximum(features_list[self.idx_max], mean_val)
                
        return tf.concat(features_list, axis=1)

def get_feature_indices():
    try: idx_min = SHAP_FEATURES.index('Bwd Packet Length Min')
    except: idx_min = None
    try: idx_mean = SHAP_FEATURES.index('Bwd Packet Length Mean')
    except: idx_mean = None
    try: idx_max = SHAP_FEATURES.index('Bwd Packet Length Max')
    except: idx_max = None
    return idx_min, idx_mean, idx_max

def construir_generador(input_dim, limites_tensor):
    idx_min, idx_mean, idx_max = get_feature_indices()
    inp = layers.Input(shape=(input_dim,), name="Input_Attack")
    x = layers.Dense(64, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    delta_raw = layers.Dense(input_dim, activation='tanh')(x)
    delta_scaled = layers.Multiply(name="Escalar_Limites")([delta_raw, limites_tensor])
    modified = layers.Add()([inp, delta_scaled])
    final_output = HierarchicalLogicLayer(idx_min, idx_mean, idx_max)([inp, modified])
    final_output = layers.Lambda(lambda z: K.clip(z, 0.0, 1.0))(final_output)
    return models.Model(inp, final_output)

def construir_discriminador(input_dim):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inp, out)

# ==========================================
# 3. ENTRENAMIENTO (SIN PHYSICS LOSS, CON CHECKPOINTING)
# ==========================================
def entrenar_gan(df_benign, df_attacks_shap, df_attacks_full_orig):
    input_dim = len(SHAP_FEATURES)
    scaler = MinMaxScaler()
    combined = pd.concat([df_benign, df_attacks_shap], ignore_index=True)
    scaler.fit(combined)
    
    X_benign = scaler.transform(df_benign)
    X_attacks = scaler.transform(df_attacks_shap)
    
    data_ranges = scaler.data_range_
    data_ranges[data_ranges == 0] = 1.0
    limits_abs = np.array([LIMITES_FISICOS[col] for col in SHAP_FEATURES])
    tf_limits = K.constant(limits_abs / data_ranges, shape=(1, input_dim))
    
    generator = construir_generador(input_dim, tf_limits)
    discriminator = construir_discriminador(input_dim)
    
    g_opt = optimizers.Adam(LEARNING_RATE)
    d_opt = optimizers.Adam(LEARNING_RATE)
    loss_bc = tf.keras.losses.BinaryCrossentropy()
    loss_mse = tf.keras.losses.MeanSquaredError()
    
    # Preparar RF
    rf_model = None
    rf_columns = []
    try:
        rf_model = joblib.load(MODELO_VICTIMA_PATH)
        rf_columns = list(rf_model.feature_names_in_)
        print(f"[*] Auditoría RF activada. Esperando {len(rf_columns)} columnas.")
    except: pass

    # Contexto para auditoría
    audit_size = len(X_attacks)
    X_audit_raw = X_attacks[:audit_size]
    df_audit_full_context = df_attacks_full_orig.iloc[:audit_size].reset_index(drop=True)

    best_evasion = 0.0
    best_weights = generator.get_weights()
    
    print(f"[*] Entrenando GAN ({EPOCHS} épocas)...")
    
    for epoch in range(EPOCHS):
        # Bucle de entrenamiento
        idx_b = np.random.randint(0, X_benign.shape[0], BATCH_SIZE)
        idx_a = np.random.randint(0, X_attacks.shape[0], BATCH_SIZE)
        
        fake_attack = generator(X_attacks[idx_a])
        with tf.GradientTape() as tape:
            d_loss = loss_bc(tf.ones_like(discriminator(X_benign[idx_b])), discriminator(X_benign[idx_b])) + \
                     loss_bc(tf.zeros_like(discriminator(fake_attack)), discriminator(fake_attack))
        d_opt.apply_gradients(zip(tape.gradient(d_loss, discriminator.trainable_variables), discriminator.trainable_variables))
        
        with tf.GradientTape() as tape:
            fake_pred = generator(X_attacks[idx_a])
            total_g_loss = loss_bc(tf.ones_like(discriminator(fake_pred)), discriminator(fake_pred)) + \
                           (FACTOR_PENALIZACION * loss_mse(X_attacks[idx_a], fake_pred))
        g_opt.apply_gradients(zip(tape.gradient(total_g_loss, generator.trainable_variables), generator.trainable_variables))
        
        # --- AUDITORÍA CADA 20 EPOCHS ---
        if epoch % 20 == 0 and rf_model:
            # 1. Generar (SIN RESTRICCIONES EXTRA)
            X_gen_audit = generator.predict(X_audit_raw, verbose=0)
            vals_gen = scaler.inverse_transform(X_gen_audit)
            df_gen = pd.DataFrame(vals_gen, columns=SHAP_FEATURES)
            
            # 2. Incrustar en contexto real
            X_rf_audit = df_audit_full_context.copy()
            for col in SHAP_FEATURES:
                X_rf_audit[col] = df_gen[col].values
            
            # 3. Filtrar columnas
            X_rf_final = pd.DataFrame(index=X_rf_audit.index)
            for col in rf_columns:
                X_rf_final[col] = X_rf_audit[col] if col in X_rf_audit else 0
            
            # 4. SANITIZACIÓN ANTI-CRASH
            X_rf_final = X_rf_final.replace([np.inf, -np.inf], 1e9).fillna(0)
            X_rf_final = X_rf_final.astype(np.float32)

            try:
                preds = rf_model.predict(X_rf_final)
                try: benign_lbl = joblib.load(ENCODER_PATH).transform(['BENIGN'])[0]
                except: benign_lbl = 0
                evasion = (preds == benign_lbl).mean() * 100
                
                print(f"Ep {epoch} | Loss: {total_g_loss:.4f} | Evasión (Libre): {evasion:.1f}%")
                
                if evasion >= best_evasion:
                    best_evasion = evasion
                    best_weights = generator.get_weights()
            except Exception as e:
                print(f"Error en auditoría: {e}")

    print(f"[*] Restaurando MEJOR modelo (Evasión: {best_evasion:.1f}%)...")
    generator.set_weights(best_weights)
    return generator, scaler, X_attacks

# ==========================================
# 4. RECONSTRUCCIÓN FINAL
# ==========================================
def reparar_y_reconstruir(df_full_original, df_shap_original, df_shap_modificado):
    print("[*] Aplicando limpieza y CÁLCULO INVERSO de paquetes...")
    
    # 1. LIMPIEZA
    df_base = df_full_original.drop(columns=columnas_basura, errors='ignore').copy().reset_index(drop=True)
    
    # 2. REPARACIÓN NUMÉRICA BÁSICA
    df_shap_fixed = df_shap_modificado.copy()
    for col in SHAP_FEATURES:
        old_val = df_shap_original[col].values
        new_val = df_shap_modificado[col].values
        delta = new_val - old_val
        limit = LIMITES_FISICOS[col]
        delta_clamped = np.clip(delta, -limit, limit)
        df_shap_fixed[col] = np.maximum(old_val + delta_clamped, 0)

    # Enteros
    cols_int = ['Init_Win_bytes_forward', 'Init_Win_bytes_backward', 
                'Bwd Packet Length Min', 'Bwd Packet Length Max', 
                'Total Length of Bwd Packets', 'Fwd Header Length', 'Max Packet Length']
    for c in cols_int:
        if c in df_shap_fixed.columns: df_shap_fixed[c] = df_shap_fixed[c].round().astype(int)

    # RESTRICCIÓN FÍSICA TCP: Window Size no puede sobrepasar los 16 bits (65535)
    if 'Init_Win_bytes_forward' in df_shap_fixed.columns:
        df_shap_fixed['Init_Win_bytes_forward'] = np.minimum(df_shap_fixed['Init_Win_bytes_forward'], 65535)
    if 'Init_Win_bytes_backward' in df_shap_fixed.columns:
        df_shap_fixed['Init_Win_bytes_backward'] = np.minimum(df_shap_fixed['Init_Win_bytes_backward'], 65535)

    # 3. LÓGICA JERÁRQUICA (Backward)
    if 'Bwd Packet Length Mean' in df_shap_fixed.columns:
        mean_val = df_shap_fixed['Bwd Packet Length Mean']
        if 'Bwd Packet Length Max' in df_shap_fixed.columns:
            df_shap_fixed['Bwd Packet Length Max'] = np.maximum(df_shap_fixed['Bwd Packet Length Max'], mean_val).astype(int)
        if 'Bwd Packet Length Min' in df_shap_fixed.columns:
            df_shap_fixed['Bwd Packet Length Min'] = np.minimum(df_shap_fixed['Bwd Packet Length Min'], mean_val).astype(int)

    # 4. INJERTO
    for col in SHAP_FEATURES:
        df_base[col] = df_shap_fixed[col].values

    # 5. CÁLCULO INVERSO (Solo al final, para asegurar coherencia física)
    
    # --- A) Coherencia Backward ---
    if 'Total Length of Bwd Packets' in df_base.columns and 'Bwd Packet Length Mean' in df_base.columns:
        gan_total_len = df_base['Total Length of Bwd Packets']
        gan_mean = df_base['Bwd Packet Length Mean']
        gan_mean_safe = gan_mean.replace(0, 1) 
        new_packet_count = np.round(gan_total_len / gan_mean_safe).astype(int)
        new_packet_count = np.maximum(new_packet_count, 1)
        
        mask_zero_len = (gan_total_len == 0)
        if 'Total Backward Packets' in df_base.columns:
             original_count = df_base['Total Backward Packets']
             new_packet_count = np.where(mask_zero_len, original_count, new_packet_count)
             print("    -> Recalculando 'Total Backward Packets' para consistencia matemática...")
             df_base['Total Backward Packets'] = new_packet_count

    # --- B) Coherencia Forward (REGLA TCP + Header/32) ---
    if 'Fwd Header Length' in df_base.columns:
        # 1. Alineación TCP: Forzar que sea múltiplo de 4
        raw_header = df_base['Fwd Header Length']
        header_aligned = np.round(raw_header / 4.0) * 4.0
        df_base['Fwd Header Length'] = header_aligned.astype(int)

        # 2. Recalcular Total Fwd Packets si es necesario
        if 'Total Fwd Packets' in df_base.columns:
            new_header_len = df_base['Fwd Header Length']
            estimated_packets = np.ceil(new_header_len / 32.0).astype(int)
            original_packets = df_base['Total Fwd Packets']
            final_packets = np.maximum(original_packets, estimated_packets)
            print("    -> Recalculando 'Total Fwd Packets' (Alineado/32) para consistencia...")
            df_base['Total Fwd Packets'] = final_packets

    # --- C) Coherencia Packet Length Mean (DESPEJAR FWD PAYLOAD Y SINCRONIZAR) ---
    if 'Packet Length Mean' in df_base.columns and 'Total Length of Fwd Packets' in df_base.columns:
        
        gan_mean = df_base['Packet Length Mean']
        
        bwd_len = df_base['Total Length of Bwd Packets'] if 'Total Length of Bwd Packets' in df_base.columns else 0
        bwd_pkts = df_base['Total Backward Packets'] if 'Total Backward Packets' in df_base.columns else 0
        fwd_pkts = df_base['Total Fwd Packets'] if 'Total Fwd Packets' in df_base.columns else 1
        total_pkts = fwd_pkts + bwd_pkts
        
        new_fwd_len = (gan_mean * total_pkts) - bwd_len
        new_fwd_len = np.maximum(new_fwd_len, 0)
        max_fwd_len_allowed = fwd_pkts * 1460
        new_fwd_len = np.minimum(new_fwd_len, max_fwd_len_allowed)
        
        df_base['Total Length of Fwd Packets'] = np.round(new_fwd_len).astype(int)
        df_base['Packet Length Mean'] = (df_base['Total Length of Fwd Packets'] + bwd_len) / np.maximum(total_pkts, 1)
        
        if 'Fwd Packet Length Mean' in df_base.columns:
            df_base['Fwd Packet Length Mean'] = df_base['Total Length of Fwd Packets'] / np.maximum(fwd_pkts, 1)
        if 'Avg Fwd Segment Size' in df_base.columns:
            df_base['Avg Fwd Segment Size'] = df_base['Fwd Packet Length Mean']
        if 'Average Packet Size' in df_base.columns:
            df_base['Average Packet Size'] = df_base['Packet Length Mean']
        if 'Fwd Packet Length Max' in df_base.columns and 'Fwd Packet Length Mean' in df_base.columns:
            df_base['Fwd Packet Length Max'] = np.maximum(df_base['Fwd Packet Length Max'], df_base['Fwd Packet Length Mean']).astype(int)
        if 'Max Packet Length' in df_base.columns and 'Fwd Packet Length Max' in df_base.columns:
            df_base['Max Packet Length'] = np.maximum(df_base['Max Packet Length'], df_base['Fwd Packet Length Max']).astype(int)
            
        print("    -> [PAYLOAD] Matemáticas de Payload y columnas hermanas sincronizadas 100%.")

    # --- D) Coherencia Flow Duration e IAT (MODELO PING-PONG / REALISTA) ---
    if 'Flow IAT Mean' in df_base.columns:
        total_pkts = df_base['Total Fwd Packets'] + df_base['Total Backward Packets']
        intervalos  = np.maximum(total_pkts - 1, 1)
        fwd_intervalos = np.maximum(df_base['Total Fwd Packets'] - 1, 0)
        bwd_intervalos = np.maximum(df_base['Total Backward Packets'] - 1, 0)

        # 1. Calculamos la duración teórica que quiere la GAN
        duracion_teorica = df_base['Flow IAT Mean'] * intervalos
        
        # 2. Forzamos el límite estricto de CICFlowMeter (Topamos a 119.9 segundos)
        df_base['Flow Duration'] = np.minimum(duracion_teorica, 119990000).round().astype(int)
        
        # 3. Recalculamos el Flow IAT Mean real para que cuadre perfectamente con el tope
        df_base['Flow IAT Mean'] = df_base['Flow Duration'] / intervalos

        # 4. LA MAGIA DEL PING-PONG (Tiempos Concurrentes, no sumados)
        # Rescatamos los valores que pidió la GAN originalmente
        fwd_raw = df_shap_fixed['Fwd IAT Total'] if 'Fwd IAT Total' in df_shap_fixed.columns else df_base['Fwd IAT Total']
        bwd_raw = df_shap_fixed['Bwd IAT Total'] if 'Bwd IAT Total' in df_shap_fixed.columns else df_base['Bwd IAT Total']

        # En TCP real entrelazado, el Flow Duration lo marca la dirección que más tarda.
        # Buscamos quién es el más lento (Max) y sacamos una proporción para escalar.
        max_raw = np.maximum(np.maximum(fwd_raw, bwd_raw), 1)
        ratio_escala = df_base['Flow Duration'] / max_raw

        # Escalamos ambos cronómetros (Ahora el mayor será EXACTAMENTE igual al Flow Duration)
        fwd_iat_total_nuevo = (fwd_raw * ratio_escala).round().astype(int)
        bwd_iat_total_nuevo = (bwd_raw * ratio_escala).round().astype(int)

        if 'Fwd IAT Total' in df_base.columns:
            df_base['Fwd IAT Total'] = fwd_iat_total_nuevo
        if 'Bwd IAT Total' in df_base.columns:
            df_base['Bwd IAT Total'] = bwd_iat_total_nuevo

        # 5. Recalcular las medias direccionales
        fwd_iat_mean_nuevo = fwd_iat_total_nuevo / np.maximum(fwd_intervalos, 1)
        bwd_iat_mean_nuevo = bwd_iat_total_nuevo / np.maximum(bwd_intervalos, 1)

        # 6. Actualizar FWD IAT (Std, Mean, Max, Min) con redondeos seguros
        if 'Fwd IAT Std' in df_base.columns and 'Fwd IAT Mean' in df_base.columns:
            old_mean_fwd = df_base['Fwd IAT Mean'].replace(0, 1)
            ratio_fwd = fwd_iat_mean_nuevo / old_mean_fwd
            df_base['Fwd IAT Std'] = (df_base['Fwd IAT Std'] * ratio_fwd).clip(lower=0).round(2)
            
        if 'Fwd IAT Mean' in df_base.columns:
            df_base['Fwd IAT Mean'] = fwd_iat_mean_nuevo.round(2)
        if 'Fwd IAT Max' in df_base.columns:
            df_base['Fwd IAT Max'] = np.maximum(df_base['Fwd IAT Max'], np.ceil(fwd_iat_mean_nuevo)).astype(int)
        if 'Fwd IAT Min' in df_base.columns:
            df_base['Fwd IAT Min'] = np.minimum(df_base['Fwd IAT Min'], np.floor(fwd_iat_mean_nuevo)).astype(int)

        # 7. Actualizar BWD IAT (Std, Mean, Max, Min) con redondeos seguros
        if 'Bwd IAT Std' in df_base.columns and 'Bwd IAT Mean' in df_base.columns:
            old_mean_bwd = df_base['Bwd IAT Mean'].replace(0, 1)
            ratio_bwd = bwd_iat_mean_nuevo / old_mean_bwd
            df_base['Bwd IAT Std'] = (df_base['Bwd IAT Std'] * ratio_bwd).clip(lower=0).round(2)
            
        if 'Bwd IAT Mean' in df_base.columns:
            df_base['Bwd IAT Mean'] = bwd_iat_mean_nuevo.round(2)
        if 'Bwd IAT Max' in df_base.columns:
            df_base['Bwd IAT Max'] = np.maximum(df_base['Bwd IAT Max'], np.ceil(bwd_iat_mean_nuevo)).astype(int)
        if 'Bwd IAT Min' in df_base.columns:
            df_base['Bwd IAT Min'] = np.minimum(df_base['Bwd IAT Min'], np.floor(bwd_iat_mean_nuevo)).astype(int)

        print("    -> [IAT] Modelo PING-PONG activado: Flow Duration dictado por el máximo concurrente (NO por la suma).")

    # 6. SANITIZACIÓN FINAL Y RETURN (Indentado a la altura del IF)
    df_base = df_base.replace([np.inf, -np.inf], 1e9).fillna(0)
    return df_base, df_shap_fixed

# ==========================================
# 5. VALIDACIÓN RF
# ==========================================
def validar_rf(df_final_limpio):
    print("\n[*] Validando eficacia contra Random Forest...")
    try:
        model = joblib.load(MODELO_VICTIMA_PATH)
        try:
            encoder = joblib.load(ENCODER_PATH)
            benign_lbl = encoder.transform(['BENIGN'])[0]
        except: benign_lbl = 0
            
        model_cols = model.feature_names_in_
        X_test = pd.DataFrame(index=df_final_limpio.index)
        for col in model_cols:
            if col in df_final_limpio.columns:
                X_test[col] = df_final_limpio[col]
            else:
                X_test[col] = 0 
        
        X_test = X_test.replace([np.inf, -np.inf], 1e9).fillna(0)
        X_test = X_test.astype(np.float32)

        preds = model.predict(X_test)
        success = (preds == benign_lbl)
        
        print(f"\n" + "="*40)
        print(f"[★] TASA DE EVASIÓN REAL: {success.mean()*100:.2f}%")
        print(f"    ({success.sum()} ataques ahora parecen benignos)")
        print(f"="*40)
            
    except Exception as e:
        print(f"[-] Error validación: {e}")

# ==========================================
# MAIN
# ==========================================
def main():
    df_benign_shap, df_attacks_shap_orig, df_attacks_full_orig = cargar_datos_completos()
    generator, scaler, X_raw = entrenar_gan(df_benign_shap, df_attacks_shap_orig, df_attacks_full_orig)
    
    print("[*] Generando perturbaciones finales...")
    X_adv = generator.predict(X_raw)
    
    vals_orig = scaler.inverse_transform(X_raw)
    vals_adv = scaler.inverse_transform(X_adv)
    df_shap_orig = pd.DataFrame(vals_orig, columns=SHAP_FEATURES)
    df_shap_adv_raw = pd.DataFrame(vals_adv, columns=SHAP_FEATURES)
    
    df_completo_limpio, df_shap_final = reparar_y_reconstruir(
        df_attacks_full_orig, 
        df_shap_orig, 
        df_shap_adv_raw
    )
    
    print("\n[*] Guardando resultados...")
    df_completo_limpio.to_csv("ataques_adversarios_COMPLETOS.csv", index=False)
    
    reporte = pd.DataFrame()
    reporte['Archivo'] = df_attacks_full_orig['_ORIGIN_FILE_'].values
    for col in SHAP_FEATURES:
        reporte[f'OLD_{col}'] = df_shap_orig[col].values
        reporte[f'NEW_{col}'] = df_shap_final[col].values
    reporte.to_csv("REPORTE_GAN.csv", index=False)
    
    validar_rf(df_completo_limpio)

if __name__ == "__main__":
    main()