# ==========================================
# GAN ADVERSARIA: VERSIÓN LIBRE PERO DETERMINISTA (RESULTADOS FIJOS)
# ==========================================

import os
import random
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, backend as K
from sklearn.preprocessing import MinMaxScaler
import joblib


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- BLOQUE DE REPRODUCIBILIDAD ---
SEMILLA = 42
os.environ['PYTHONHASHSEED'] = str(SEMILLA)
random.seed(SEMILLA)
np.random.seed(SEMILLA)
tf.random.set_seed(SEMILLA)
tf.config.experimental.enable_op_determinism()
print(f"[*] Modo Determinista ACTIVADO. Semilla: {SEMILLA}")

# --- CONFIGURACIÓN ---
DATOS_REALES_DIR    = 'Dataset/MachineLearningCVE'
MODELO_VICTIMA_PATH = 'random_forest_cic17_best.pkl'
ENCODER_PATH        = 'label_encoder.pkl'

CANTIDAD_ATAQUES   = 1000
EPOCHS             = 300
BATCH_SIZE         = 64
LEARNING_RATE      = 0.0001
FACTOR_PENALIZACION = 25.0

# Límites físicos (Deltas máximos permitidos por la GAN)
LIMITES_FISICOS = {
    'Init_Win_bytes_forward':   65535.0,
    'Init_Win_bytes_backward':      0.0,
    'Bwd Packet Length Min':        2.0,
    'Fwd Header Length':          400.0,
    'Bwd Packet Length Std':       10.0,
    'Bwd Packet Length Mean':       5.0,
    'Avg Bwd Segment Size':         5.0,
    'Bwd Packet Length Max':       10.0,
    'Packet Length Mean':         800.0,
    'Max Packet Length':           10.0,
    'Total Length of Bwd Packets': 50.0,
    'Flow IAT Mean':             100.0,
}

SHAP_FEATURES = list(LIMITES_FISICOS.keys())

# Columnas sin valor predictivo que se eliminan del CSV final
columnas_basura = [
    'Destination Port', 'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Timestamp',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    'URG Flag Count', 'CWE Flag Count', 'CWR Flag Count', 'ECE Flag Count',
    'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Fwd Avg Bytes/Bulk',
    'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
    'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
    'Fwd Seg Size Min', 'min_seg_size_forward', 'Fwd Header Length.1',
]


# ==========================================
# 0. FILTRO DE APLICABILIDAD
# ==========================================
def filtrar_flujos_aplicables(df):
    """
    Descarta flujos con anomalías físicas irrecuperables ANTES de pasarlos
    por la GAN, para que ésta aprenda solo sobre muestras coherentes.

    Criterios de descarte (INAPLICABLE):
      1. Flow Packets/s >= 1e9  → saturación de CICFlowMeter (div/0)
      2. Flow Duration == 0 con más de 1 paquete → flujo instantáneo imposible
      3. Flow IAT Min < 0       → bug de timestamp del extractor original
      4. Payload Fwd == 0, sin SYN y con >2 paquetes → sin contenido ni handshake
    """
    n_antes = len(df)

    mask_ok = (
        (df['Flow Packets/s'] < 1e9)
        & ~((df['Flow Duration'] == 0) & (df['Total Fwd Packets'] > 1))
        & (df['Flow IAT Min'] >= 0)
        & ~(
            (df['Total Length of Fwd Packets'] == 0)
            & (df['SYN Flag Count'] == 0)
            & (df['Total Fwd Packets'] > 2)
        )
    )

    df_filtrado = df[mask_ok].copy()
    n_despues   = len(df_filtrado)
    descartados = n_antes - n_despues
    print(f"[✓] Filtro de aplicabilidad: {n_antes} → {n_despues} flujos "
          f"({descartados} descartados, {descartados / n_antes * 100:.1f}%)")
    return df_filtrado


# ==========================================
# 1. CARGA DE DATOS
# ==========================================
def cargar_datos_completos():
    print("[*] Cargando dataset completo...")
    files   = glob.glob(os.path.join(DATOS_REALES_DIR, "*.csv"))
    df_list = []
    for f in files:
        try:
            df_t = pd.read_csv(f)
            df_t.columns = df_t.columns.str.strip()
            df_t['_ORIGIN_FILE_'] = os.path.basename(f)
            df_t['_ORIGIN_IDX_']  = df_t.index
            df_list.append(df_t)
        except:
            pass

    df_full   = pd.concat(df_list, ignore_index=True)
    label_col = [c for c in df_full.columns if 'label' in c.lower()][0]

    # ── Benignos ──────────────────────────────────────────────────────────
    df_benign_train = df_full[df_full[label_col] == 'BENIGN'].copy()
    for col in SHAP_FEATURES:
        if col in df_benign_train.columns:
            df_benign_train = df_benign_train[df_benign_train[col] >= 0]
    df_benign_shap = df_benign_train[SHAP_FEATURES].dropna()

    # ── Ataques ───────────────────────────────────────────────────────────
    df_attacks_full = df_full[df_full[label_col] != 'BENIGN'].dropna()

    # FILTRO DE APLICABILIDAD: solo flujos físicamente coherentes
    df_attacks_full = filtrar_flujos_aplicables(df_attacks_full)

    # ── MUESTREO: mínimo 30 por tipo, hasta CANTIDAD_ATAQUES total ────────
    MIN_POR_TIPO = 30

    # 1. Garantizar mínimo 30 de cada tipo (con reemplazo si hacen falta)
    partes = []
    for tipo, grupo in df_attacks_full.groupby(label_col):
        if len(grupo) >= MIN_POR_TIPO:
            partes.append(grupo.sample(n=MIN_POR_TIPO, random_state=SEMILLA))
        else:
            partes.append(grupo.sample(n=MIN_POR_TIPO, random_state=SEMILLA, replace=True))
            print(f"    [!] '{tipo}': solo {len(grupo)} muestras → rellenado con repetición hasta {MIN_POR_TIPO}")

    df_obligatorio    = pd.concat(partes, ignore_index=True)
    cuota_obligatoria = len(df_obligatorio)
    print(f"[✓] Cuota mínima garantizada: {cuota_obligatoria} filas ({len(partes)} tipos × {MIN_POR_TIPO})")

    # 2. Rellenar hasta CANTIDAD_ATAQUES con el resto disponible (o aleatorio con reemplazo)
    restantes_necesarios = CANTIDAD_ATAQUES - cuota_obligatoria

    if restantes_necesarios > 0:
        indices_usados = set(df_obligatorio.index) & set(df_attacks_full.index)
        df_pool        = df_attacks_full.drop(index=list(indices_usados), errors='ignore')

        if len(df_pool) >= restantes_necesarios:
            try:
                from sklearn.model_selection import train_test_split
                _, df_relleno = train_test_split(
                    df_pool,
                    test_size=restantes_necesarios,
                    random_state=SEMILLA,
                    stratify=df_pool[label_col]
                )
            except ValueError:
                df_relleno = df_pool.sample(n=restantes_necesarios, random_state=SEMILLA)
        else:
            print(f"    [!] Pool insuficiente ({len(df_pool)}) → relleno aleatorio con reemplazo")
            df_relleno = df_attacks_full.sample(n=restantes_necesarios, random_state=SEMILLA, replace=True)

        df_attacks_full = pd.concat([df_obligatorio, df_relleno], ignore_index=True)
    else:
        df_attacks_full = df_obligatorio

    # Mezclar para que no queden agrupados por tipo
    df_attacks_full = df_attacks_full.sample(frac=1, random_state=SEMILLA).reset_index(drop=True)

    print(f"[✓] Dataset final de ataques: {len(df_attacks_full)} filas")
    tipos_finales = df_attacks_full[label_col].value_counts()
    print(tipos_finales.to_string())

    df_attacks_shap = df_attacks_full[SHAP_FEATURES].copy()

    print(f"[✓] Benignos: {len(df_benign_shap)} | Ataques: {len(df_attacks_full)}")
    return df_benign_shap, df_attacks_shap, df_attacks_full


# ==========================================
# 2. GAN (Cerebro)
# ==========================================
class HierarchicalLogicLayer(layers.Layer):
    def __init__(self, idx_min, idx_mean, idx_max, **kwargs):
        super(HierarchicalLogicLayer, self).__init__(**kwargs)
        self.idx_min  = idx_min
        self.idx_mean = idx_mean
        self.idx_max  = idx_max

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
    try:    idx_min  = SHAP_FEATURES.index('Bwd Packet Length Min')
    except: idx_min  = None
    try:    idx_mean = SHAP_FEATURES.index('Bwd Packet Length Mean')
    except: idx_mean = None
    try:    idx_max  = SHAP_FEATURES.index('Bwd Packet Length Max')
    except: idx_max  = None
    return idx_min, idx_mean, idx_max


def construir_generador(input_dim, limites_tensor):
    idx_min, idx_mean, idx_max = get_feature_indices()
    inp          = layers.Input(shape=(input_dim,), name="Input_Attack")
    x            = layers.Dense(64, activation='relu')(inp)
    x            = layers.BatchNormalization()(x)
    x            = layers.Dense(64, activation='relu')(x)
    delta_raw    = layers.Dense(input_dim, activation='tanh')(x)
    delta_scaled = layers.Multiply(name="Escalar_Limites")([delta_raw, limites_tensor])
    modified     = layers.Add()([inp, delta_scaled])
    final_output = HierarchicalLogicLayer(idx_min, idx_mean, idx_max)([inp, modified])
    final_output = layers.Lambda(lambda z: K.clip(z, 0.0, 1.0))(final_output)
    return models.Model(inp, final_output)


def construir_discriminador(input_dim):
    inp = layers.Input(shape=(input_dim,))
    x   = layers.Dense(128, activation='relu')(inp)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inp, out)


# ==========================================
# 3. ENTRENAMIENTO (CON CHECKPOINTING)
# ==========================================
def entrenar_gan(df_benign, df_attacks_shap, df_attacks_full_orig):
    input_dim = len(SHAP_FEATURES)
    scaler    = MinMaxScaler()
    combined  = pd.concat([df_benign, df_attacks_shap], ignore_index=True)
    scaler.fit(combined)

    X_benign  = scaler.transform(df_benign)
    X_attacks = scaler.transform(df_attacks_shap)

    data_ranges = scaler.data_range_.copy()
    data_ranges[data_ranges == 0] = 1.0
    limits_abs = np.array([LIMITES_FISICOS[col] for col in SHAP_FEATURES])
    tf_limits  = K.constant(limits_abs / data_ranges, shape=(1, input_dim))

    generator     = construir_generador(input_dim, tf_limits)
    discriminator = construir_discriminador(input_dim)

    g_opt    = optimizers.Adam(LEARNING_RATE)
    d_opt    = optimizers.Adam(LEARNING_RATE)
    loss_bc  = tf.keras.losses.BinaryCrossentropy()
    loss_mse = tf.keras.losses.MeanSquaredError()

    # Preparar modelo víctima (RF)
    rf_model   = None
    rf_columns = []
    try:
        rf_model   = joblib.load(MODELO_VICTIMA_PATH)
        rf_columns = list(rf_model.feature_names_in_)
        print(f"[*] Auditoría RF activada. Esperando {len(rf_columns)} columnas.")
    except:
        pass

    # Contexto para auditoría
    audit_size            = len(X_attacks)
    X_audit_raw           = X_attacks[:audit_size]
    df_audit_full_context = df_attacks_full_orig.iloc[:audit_size].reset_index(drop=True)

    best_evasion = 0.0
    best_weights = generator.get_weights()

    print(f"[*] Entrenando GAN ({EPOCHS} épocas)...")

    for epoch in range(EPOCHS):
        idx_b = np.random.randint(0, X_benign.shape[0], BATCH_SIZE)
        idx_a = np.random.randint(0, X_attacks.shape[0], BATCH_SIZE)

        fake_attack = generator(X_attacks[idx_a])
        with tf.GradientTape() as tape:
            d_loss = (
                loss_bc(tf.ones_like(discriminator(X_benign[idx_b])),  discriminator(X_benign[idx_b]))
                + loss_bc(tf.zeros_like(discriminator(fake_attack)), discriminator(fake_attack))
            )
        d_opt.apply_gradients(
            zip(tape.gradient(d_loss, discriminator.trainable_variables),
                discriminator.trainable_variables)
        )

        with tf.GradientTape() as tape:
            fake_pred    = generator(X_attacks[idx_a])
            total_g_loss = (
                loss_bc(tf.ones_like(discriminator(fake_pred)), discriminator(fake_pred))
                + FACTOR_PENALIZACION * loss_mse(X_attacks[idx_a], fake_pred)
            )
        g_opt.apply_gradients(
            zip(tape.gradient(total_g_loss, generator.trainable_variables),
                generator.trainable_variables)
        )

        # --- AUDITORÍA CADA 20 EPOCHS ---
        if epoch % 20 == 0 and rf_model:
            X_gen_audit = generator.predict(X_audit_raw, verbose=0)
            vals_gen    = scaler.inverse_transform(X_gen_audit)
            df_gen      = pd.DataFrame(vals_gen, columns=SHAP_FEATURES)

            X_rf_audit = df_audit_full_context.copy()
            for col in SHAP_FEATURES:
                X_rf_audit[col] = df_gen[col].values

            X_rf_final = pd.DataFrame(index=X_rf_audit.index)
            for col in rf_columns:
                X_rf_final[col] = X_rf_audit[col] if col in X_rf_audit else 0

            X_rf_final = X_rf_final.replace([np.inf, -np.inf], 1e9).fillna(0)
            X_rf_final = X_rf_final.astype(np.float32)

            try:
                preds = rf_model.predict(X_rf_final)
                try:    benign_lbl = joblib.load(ENCODER_PATH).transform(['BENIGN'])[0]
                except: benign_lbl = 0
                evasion = (preds == benign_lbl).mean() * 100
                print(f"Ep {epoch} | Loss: {total_g_loss:.4f} | Evasión: {evasion:.1f}%")
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
        old_val       = df_shap_original[col].values
        new_val       = df_shap_modificado[col].values
        delta         = new_val - old_val
        limit         = LIMITES_FISICOS[col]
        delta_clamped = np.clip(delta, -limit, limit)
        df_shap_fixed[col] = np.maximum(old_val + delta_clamped, 0)

    # Conversión a enteros donde aplica
    cols_int = [
        'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
        'Bwd Packet Length Min',  'Bwd Packet Length Max',
        'Total Length of Bwd Packets', 'Fwd Header Length', 'Max Packet Length',
    ]
    for c in cols_int:
        if c in df_shap_fixed.columns:
            df_shap_fixed[c] = df_shap_fixed[c].round().astype(int)

    # Restricción física TCP: Window Size ≤ 16 bits
    for win_col in ('Init_Win_bytes_forward', 'Init_Win_bytes_backward'):
        if win_col in df_shap_fixed.columns:
            df_shap_fixed[win_col] = np.minimum(df_shap_fixed[win_col], 65535)

    # 3. LÓGICA JERÁRQUICA (Backward: Min ≤ Mean ≤ Max)
    if 'Bwd Packet Length Mean' in df_shap_fixed.columns:
        mean_val = df_shap_fixed['Bwd Packet Length Mean']
        if 'Bwd Packet Length Max' in df_shap_fixed.columns:
            df_shap_fixed['Bwd Packet Length Max'] = np.maximum(
                df_shap_fixed['Bwd Packet Length Max'], mean_val).astype(int)
        if 'Bwd Packet Length Min' in df_shap_fixed.columns:
            df_shap_fixed['Bwd Packet Length Min'] = np.minimum(
                df_shap_fixed['Bwd Packet Length Min'], mean_val).astype(int)

    # 4. INJERTO en el dataframe completo
    for col in SHAP_FEATURES:
        df_base[col] = df_shap_fixed[col].values

    # 5. CÁLCULO INVERSO para coherencia física

    # --- A) Coherencia Backward ---
    if 'Total Length of Bwd Packets' in df_base.columns and 'Bwd Packet Length Mean' in df_base.columns:
        gan_total_len    = df_base['Total Length of Bwd Packets']
        gan_mean_safe    = df_base['Bwd Packet Length Mean'].replace(0, 1)
        new_packet_count = np.maximum(np.round(gan_total_len / gan_mean_safe).astype(int), 1)
        mask_zero_len    = (gan_total_len == 0)
        if 'Total Backward Packets' in df_base.columns:
            original_count   = df_base['Total Backward Packets']
            new_packet_count = np.where(mask_zero_len, original_count, new_packet_count)
            print("    -> Recalculando 'Total Backward Packets' para consistencia matemática...")
            df_base['Total Backward Packets'] = new_packet_count

    # --- B) Coherencia Forward (alineación TCP + Header/32) ---
    if 'Fwd Header Length' in df_base.columns:
        header_aligned           = np.round(df_base['Fwd Header Length'] / 4.0) * 4.0
        df_base['Fwd Header Length'] = header_aligned.astype(int)
        if 'Total Fwd Packets' in df_base.columns:
            estimated_packets        = np.ceil(df_base['Fwd Header Length'] / 32.0).astype(int)
            df_base['Total Fwd Packets'] = np.maximum(df_base['Total Fwd Packets'], estimated_packets)
            print("    -> Recalculando 'Total Fwd Packets' (Alineado/32) para consistencia...")

    # --- C) Coherencia Packet Length Mean (despejar Fwd payload y sincronizar) ---
    if 'Packet Length Mean' in df_base.columns and 'Total Length of Fwd Packets' in df_base.columns:
        gan_mean   = df_base['Packet Length Mean']
        bwd_len    = df_base.get('Total Length of Bwd Packets', 0)
        bwd_pkts   = df_base.get('Total Backward Packets', 0)
        fwd_pkts   = df_base.get('Total Fwd Packets', pd.Series([1] * len(df_base)))
        total_pkts = fwd_pkts + bwd_pkts

        new_fwd_len = np.maximum((gan_mean * total_pkts) - bwd_len, 0)
        new_fwd_len = np.minimum(new_fwd_len, fwd_pkts * 1460)

        df_base['Total Length of Fwd Packets'] = np.round(new_fwd_len).astype(int)
        df_base['Packet Length Mean'] = (
            (df_base['Total Length of Fwd Packets'] + bwd_len) / np.maximum(total_pkts, 1)
        )

        if 'Fwd Packet Length Mean' in df_base.columns:
            df_base['Fwd Packet Length Mean'] = (
                df_base['Total Length of Fwd Packets'] / np.maximum(fwd_pkts, 1)
            )
        if 'Avg Fwd Segment Size' in df_base.columns:
            df_base['Avg Fwd Segment Size'] = df_base['Fwd Packet Length Mean']
        if 'Average Packet Size' in df_base.columns:
            df_base['Average Packet Size'] = df_base['Packet Length Mean']
        if 'Fwd Packet Length Max' in df_base.columns and 'Fwd Packet Length Mean' in df_base.columns:
            df_base['Fwd Packet Length Max'] = np.maximum(
                df_base['Fwd Packet Length Max'], df_base['Fwd Packet Length Mean']).astype(int)
        if 'Max Packet Length' in df_base.columns and 'Fwd Packet Length Max' in df_base.columns:
            df_base['Max Packet Length'] = np.maximum(
                df_base['Max Packet Length'], df_base['Fwd Packet Length Max']).astype(int)

        print("    -> [PAYLOAD] Payload y columnas hermanas sincronizadas.")

    # --- D) Coherencia Flow Duration e IAT (modelo ping-pong) ---
    if 'Flow IAT Mean' in df_base.columns:
        fwd_pkts       = df_base['Total Fwd Packets']
        bwd_pkts       = df_base['Total Backward Packets']
        total_pkts     = fwd_pkts + bwd_pkts
        intervalos     = np.maximum(total_pkts - 1, 1)
        fwd_intervalos = np.maximum(fwd_pkts - 1, 0)
        bwd_intervalos = np.maximum(bwd_pkts - 1, 0)

        # Escalar OLD_Flow_Duration por el ratio de cambio del IAT
        # Evita heredar la incoherencia del dataset original
        # (CIC-IDS-2017 tiene flujos donde IAT_Mean * intervalos != Flow_Duration)
        iat_old_safe = np.maximum(df_shap_original['Flow IAT Mean'].values, 1)
        iat_new      = df_base['Flow IAT Mean'].values
        ratio_cambio = iat_new / iat_old_safe

        dur_old = (df_shap_original['Flow Duration'].values
                   if 'Flow Duration' in df_shap_original.columns
                   else df_base['Flow Duration'].values)

        duracion_ajustada        = dur_old * ratio_cambio
        df_base['Flow Duration'] = np.minimum(
            duracion_ajustada, 119990000
        ).round().astype(int)

        # IAT Mean recalculado para coherencia interna perfecta con la duración
        df_base['Flow IAT Mean'] = df_base['Flow Duration'] / intervalos

        # Ping-pong: escalar Fwd/Bwd IAT Total al Flow Duration
        fwd_raw = (df_shap_fixed['Fwd IAT Total']
                   if 'Fwd IAT Total' in df_shap_fixed.columns
                   else df_base['Fwd IAT Total'])
        bwd_raw = (df_shap_fixed['Bwd IAT Total']
                   if 'Bwd IAT Total' in df_shap_fixed.columns
                   else df_base['Bwd IAT Total'])

        max_raw      = np.maximum(np.maximum(fwd_raw, bwd_raw), 1)
        ratio_escala = df_base['Flow Duration'] / max_raw

        fwd_iat_total_nuevo = (fwd_raw * ratio_escala).round().astype(int)
        bwd_iat_total_nuevo = (bwd_raw * ratio_escala).round().astype(int)

        if 'Fwd IAT Total' in df_base.columns:
            df_base['Fwd IAT Total'] = fwd_iat_total_nuevo
        if 'Bwd IAT Total' in df_base.columns:
            df_base['Bwd IAT Total'] = bwd_iat_total_nuevo

        fwd_iat_mean_nuevo = fwd_iat_total_nuevo / np.maximum(fwd_intervalos, 1)
        bwd_iat_mean_nuevo = bwd_iat_total_nuevo / np.maximum(bwd_intervalos, 1)

        if 'Fwd IAT Std' in df_base.columns and 'Fwd IAT Mean' in df_base.columns:
            ratio_fwd = fwd_iat_mean_nuevo / df_base['Fwd IAT Mean'].replace(0, 1)
            df_base['Fwd IAT Std'] = (
                df_base['Fwd IAT Std'] * ratio_fwd).clip(lower=0).round(2)
        if 'Fwd IAT Mean' in df_base.columns:
            df_base['Fwd IAT Mean'] = fwd_iat_mean_nuevo.round(2)
        if 'Fwd IAT Max' in df_base.columns:
            df_base['Fwd IAT Max'] = np.maximum(
                df_base['Fwd IAT Max'], np.ceil(fwd_iat_mean_nuevo)).astype(int)
        if 'Fwd IAT Min' in df_base.columns:
            df_base['Fwd IAT Min'] = np.minimum(
                df_base['Fwd IAT Min'], np.floor(fwd_iat_mean_nuevo)).astype(int)

        if 'Bwd IAT Std' in df_base.columns and 'Bwd IAT Mean' in df_base.columns:
            ratio_bwd = bwd_iat_mean_nuevo / df_base['Bwd IAT Mean'].replace(0, 1)
            df_base['Bwd IAT Std'] = (
                df_base['Bwd IAT Std'] * ratio_bwd).clip(lower=0).round(2)
        if 'Bwd IAT Mean' in df_base.columns:
            df_base['Bwd IAT Mean'] = bwd_iat_mean_nuevo.round(2)
        if 'Bwd IAT Max' in df_base.columns:
            df_base['Bwd IAT Max'] = np.maximum(
                df_base['Bwd IAT Max'], np.ceil(bwd_iat_mean_nuevo)).astype(int)
        if 'Bwd IAT Min' in df_base.columns:
            df_base['Bwd IAT Min'] = np.minimum(
                df_base['Bwd IAT Min'], np.floor(bwd_iat_mean_nuevo)).astype(int)

        print("    -> [IAT] Flow Duration escalado desde original × ratio_IAT "
              "(sin herencia de incoherencias del dataset).")

    # --- E) RECALCULAR VELOCIDADES ---
    # Flow Packets/s = Total Packets / (Flow Duration en segundos)
    # Flow Bytes/s   = Total Bytes   / (Flow Duration en segundos)
    if 'Flow Duration' in df_base.columns:
        duracion_segundos = df_base['Flow Duration'] / 1_000_000.0
        duracion_segundos = np.maximum(duracion_segundos, 1e-6)  # evitar div/0

        fwd_pkts    = df_base.get('Total Fwd Packets',             0)
        bwd_pkts    = df_base.get('Total Backward Packets',        0)
        total_pkts  = fwd_pkts + bwd_pkts
        fwd_bytes   = df_base.get('Total Length of Fwd Packets',   0)
        bwd_bytes   = df_base.get('Total Length of Bwd Packets',   0)
        total_bytes = fwd_bytes + bwd_bytes

        if 'Flow Packets/s' in df_base.columns:
            df_base['Flow Packets/s'] = total_pkts  / duracion_segundos
        if 'Flow Bytes/s' in df_base.columns:
            df_base['Flow Bytes/s']   = total_bytes / duracion_segundos
        if 'Fwd Packets/s' in df_base.columns:
            df_base['Fwd Packets/s']  = fwd_pkts    / duracion_segundos
        if 'Bwd Packets/s' in df_base.columns:
            df_base['Bwd Packets/s']  = bwd_pkts    / duracion_segundos

        print("    -> [SPEED] Velocidades (Bytes/s, Pkts/s) recalculadas para coherencia total.")

    # --- F) GARANTÍA FINAL: Fwd Min ≤ Mean ≤ Max  ← FIX ---
    # El bloque C recalcula Fwd Packet Length Mean a partir de bytes/paquetes,
    # lo que puede dejar Mean > Max o Min > Mean si los recálculos en cascada
    # no convergen. Este bloque se ejecuta incondicionalmente al final para
    # asegurar la invariante, igual que ya se hace para la dirección Backward
    # en el paso 3 de este mismo función.
    if 'Fwd Packet Length Mean' in df_base.columns:
        fwd_mean = df_base['Fwd Packet Length Mean']
        if 'Fwd Packet Length Max' in df_base.columns:
            df_base['Fwd Packet Length Max'] = np.maximum(
                df_base['Fwd Packet Length Max'], fwd_mean).astype(int)
        if 'Fwd Packet Length Min' in df_base.columns:
            df_base['Fwd Packet Length Min'] = np.minimum(
                df_base['Fwd Packet Length Min'], fwd_mean).astype(int)
        # Propagar Max Packet Length si quedó por debajo del nuevo Fwd Max
        if 'Max Packet Length' in df_base.columns and 'Fwd Packet Length Max' in df_base.columns:
            df_base['Max Packet Length'] = np.maximum(
                df_base['Max Packet Length'], df_base['Fwd Packet Length Max']).astype(int)
        print("    -> [FWD HIER] Invariante Fwd Min ≤ Mean ≤ Max garantizada.")

    # 6. SANITIZACIÓN FINAL
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
            encoder    = joblib.load(ENCODER_PATH)
            benign_lbl = encoder.transform(['BENIGN'])[0]
        except:
            benign_lbl = 0

        model_cols = model.feature_names_in_
        X_test     = pd.DataFrame(index=df_final_limpio.index)
        for col in model_cols:
            X_test[col] = df_final_limpio[col] if col in df_final_limpio.columns else 0

        X_test  = X_test.replace([np.inf, -np.inf], 1e9).fillna(0).astype(np.float32)
        preds   = model.predict(X_test)
        success = (preds == benign_lbl)

        print("\n" + "=" * 40)
        print(f"[★] TASA DE EVASIÓN REAL: {success.mean() * 100:.2f}%")
        print(f"    ({success.sum()} ataques ahora parecen benignos)")
        print("=" * 40)

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

    vals_orig       = scaler.inverse_transform(X_raw)
    vals_adv        = scaler.inverse_transform(X_adv)
    df_shap_orig    = pd.DataFrame(vals_orig, columns=SHAP_FEATURES)
    df_shap_adv_raw = pd.DataFrame(vals_adv,  columns=SHAP_FEATURES)

    df_completo_limpio, df_shap_final = reparar_y_reconstruir(
        df_attacks_full_orig,
        df_shap_orig,
        df_shap_adv_raw,
    )

    print("\n[*] Guardando resultados...")
    df_completo_limpio.to_csv("ataques_adversarios_COMPLETOS.csv", index=False)

    reporte = pd.DataFrame()
    reporte['Archivo'] = df_attacks_full_orig['_ORIGIN_FILE_'].values

    # Recorremos TODAS las columnas del dataset final limpio
    for col in df_completo_limpio.columns:
        if col in ['_ORIGIN_FILE_', '_ORIGIN_IDX_']:
            continue
        reporte[f'OLD_{col}'] = df_attacks_full_orig[col].values if col in df_attacks_full_orig.columns else 0
        reporte[f'NEW_{col}'] = df_completo_limpio[col].values

    reporte.to_csv("REPORTE_GAN_COMPLETO.csv", index=False)
    print(f"[✓] Reporte detallado guardado en: REPORTE_GAN_COMPLETO.csv (Con todas las columnas)")

    validar_rf(df_completo_limpio)


if __name__ == "__main__":
    main()