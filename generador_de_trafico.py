import pandas as pd
import numpy as np
import os
import time
from scapy.all import IP, TCP, Raw, wrpcap

# ==========================================
# CONFIGURACIÓN
# ==========================================
ARCHIVO_CSV = 'ataques_adversarios_COMPLETOS.csv'

IP_ATACANTE     = "192.168.1.50"
PUERTO_ATACANTE = 54321
IP_VICTIMA      = "10.0.0.200"
PUERTO_VICTIMA  = 80

# ==========================================
# CARGA DEL FLUJO
# ==========================================
def seleccionar_fila_aleatoria(archivo):
    if not os.path.exists(archivo):
        print(f"[-] Error: No se encontró '{archivo}'.")
        return None, "Desconocido"
    try:
        df = pd.read_csv(archivo)
        fila = df.sample(n=1)
        nombre_ataque = fila['Label'].values[0] if 'Label' in fila.columns else "Desconocido"
        nombre_ataque = nombre_ataque.encode('ascii', 'ignore').decode().strip()
        fila_limpia = fila.drop(columns=['Label', '_ORIGIN_FILE_', '_ORIGIN_IDX_'], errors='ignore')
        print(f"\n[✓] Fila seleccionada (Ataque: {nombre_ataque})")
        print("-" * 50)
        print(fila_limpia.iloc[0])
        print("-" * 50)
        return fila_limpia.iloc[0], nombre_ataque
    except Exception as e:
        print(f"[-] Error: {e}")
        return None, "Desconocido"

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================
def inferir_opciones_tcp(header_total, n_pkts):
    if n_pkts <= 0: return []
    header_medio = header_total / n_pkts
    opciones = []
    if header_medio >= 24: opciones.append(('NOP', None))
    if header_medio >= 28: opciones.append(('MSS', 1460))
    if header_medio >= 32: opciones.append(('WScale', 7))
    opciones.append(('SAckOK', b''))  # siempre presente → elimina nota azul de Wireshark
    if header_medio >= 44:
        t = int(time.time()) & 0xFFFFFFFF
        opciones.append(('Timestamp', (t, 0)))
    return opciones

def generar_tamanios(n_pkts, total_bytes, mean, std):
    if n_pkts <= 0: return []
    if total_bytes <= 0: return [0] * n_pkts
    if std <= 0 or n_pkts == 1:
        base  = total_bytes // n_pkts
        resto = total_bytes  % n_pkts
        t = [base] * n_pkts
        for i in range(resto): t[i] += 1
        return t
    t = np.random.normal(mean, std, n_pkts)
    t = np.clip(t, 0, None)
    s = t.sum()
    if s > 0: t = t * (total_bytes / s)
    t = np.round(t).astype(int)
    t[-1] = max(0, t[-1] + (total_bytes - t.sum()))
    return t.tolist()

def generar_iats_direccionales(n, mean_us, std_us):
    if n <= 0: return []
    if std_us <= 0: return [max(mean_us / 1_000_000.0, 1e-6)] * n
    iats = np.random.normal(mean_us, std_us, n)
    iats = np.clip(iats, 1, None)
    return (iats / 1_000_000.0).tolist()

def construir_listas_flags_perfectas(fwd_pkts, bwd_pkts, syn, fin, rst, psh, ack):
    """
    Construye flags TCP con handshake real dentro del límite de paquetes del CSV.
    El handshake ocupa posiciones fijas: fwd[0]=SYN, bwd[0]=SYN-ACK, fwd[1]=ACK.
    El cierre ocupa las últimas posiciones si hay FIN en el CSV.
    El resto son datos con PSH/ACK según los contadores.
    """
    flags_fwd = ['A'] * fwd_pkts
    flags_bwd = ['A'] * bwd_pkts

    # ── 1. HANDSHAKE (solo si hay paquetes suficientes) ──────────────
    handshake_fwd = 0
    handshake_bwd = 0
    if fwd_pkts >= 2 and bwd_pkts >= 1:
        flags_fwd[0] = 'S'    # SYN
        flags_bwd[0] = 'SA'   # SYN-ACK
        flags_fwd[1] = 'A'    # ACK
        handshake_fwd = 2
        handshake_bwd = 1
    elif fwd_pkts >= 1:
        # Flujo muy corto — solo SYN sin respuesta
        flags_fwd[0] = 'S'
        handshake_fwd = 1

    # ── 2. CIERRE (solo si el CSV indica FIN y hay paquetes) ─────────
    fin_count = int(fin)
    if fin_count > 0:
        if fwd_pkts > handshake_fwd:
            flags_fwd[-1] = 'FA'
        if bwd_pkts > handshake_bwd:
            flags_bwd[-1] = 'FA'

    # ── 3. PSH en paquetes de datos (entre handshake y cierre) ───────
    # Zona de datos fwd: desde handshake_fwd hasta -1 (o fin si hay cierre)
    fin_offset_fwd = 1 if (fin_count > 0 and fwd_pkts > handshake_fwd) else 0
    zona_datos_fwd = list(range(handshake_fwd, fwd_pkts - fin_offset_fwd))

    restantes_psh = int(psh)
    if zona_datos_fwd and restantes_psh > 0:
        intervalo = max(len(zona_datos_fwd) // (restantes_psh + 1), 1)
        for i, pos in enumerate(zona_datos_fwd):
            if restantes_psh <= 0:
                break
            if i % intervalo == 0:
                flags_fwd[pos] = 'PA'
                restantes_psh -= 1

    # ── 4. RST al final si el CSV lo indica ──────────────────────────
    if int(rst) > 0 and fwd_pkts > handshake_fwd:
        flags_fwd[-1] = 'R'

    return flags_fwd, flags_bwd

# ==========================================
# CONSTRUCCIÓN DEL PCAP
# ==========================================
def generar_pcap_desde_flujo(flujo, nombre_ataque):
    if flujo is None: return

    # ── Extraer métricas ─────────────────────────────────────────────
    fwd_pkts  = int(max(flujo.get('Total Fwd Packets', 1), 1))
    bwd_pkts  = int(max(flujo.get('Total Backward Packets', 0), 0))
    fwd_bytes = int(max(flujo.get('Total Length of Fwd Packets', 0), 0))
    bwd_bytes = int(max(flujo.get('Total Length of Bwd Packets', 0), 0))
    ventana_fwd = int(max(flujo.get('Init_Win_bytes_forward',  8192), 1))
    ventana_bwd = int(max(flujo.get('Init_Win_bytes_backward', 8192), 1))

    fwd_mean = float(flujo.get('Fwd Packet Length Mean', fwd_bytes / fwd_pkts if fwd_pkts else 0))
    fwd_std  = float(flujo.get('Fwd Packet Length Std',  0))
    bwd_mean = float(flujo.get('Bwd Packet Length Mean', bwd_bytes / bwd_pkts if bwd_pkts else 0))
    bwd_std  = float(flujo.get('Bwd Packet Length Std',  0))

    # Usamos Flow IAT Mean para ambas direcciones — es la única forma de que
    # la suma total de tiempos coincida con Flow Duration del CSV.
    # Fwd IAT Mean y Bwd IAT Mean son muy distintos entre sí y no reproducen
    # el Flow Duration correctamente (error medio de 54 millones de µs).
    fwd_iat_mean = float(flujo.get('Flow IAT Mean', 1000))
    fwd_iat_std  = float(flujo.get('Flow IAT Std',  0))
    bwd_iat_mean = float(flujo.get('Flow IAT Mean', 1000))
    bwd_iat_std  = float(flujo.get('Flow IAT Std',  0))

    # Flow Duration en segundos (el CSV lo da en microsegundos)
    flow_duration_us = float(flujo.get('Flow Duration', 0))
    flow_duration_s  = flow_duration_us / 1_000_000.0

    fwd_header_total = float(flujo.get('Fwd Header Length', 20 * fwd_pkts))
    bwd_header_total = float(flujo.get('Bwd Header Length', 20 * bwd_pkts))
    opciones_fwd = inferir_opciones_tcp(fwd_header_total, fwd_pkts)
    opciones_bwd = inferir_opciones_tcp(bwd_header_total, bwd_pkts)

    syn = flujo.get('SYN Flag Count', 0)
    fin = flujo.get('FIN Flag Count', 0)
    rst = flujo.get('RST Flag Count', 0)
    psh = flujo.get('PSH Flag Count', 0)
    ack = flujo.get('ACK Flag Count', 0)

    # Tamaños: los paquetes de handshake no llevan payload,
    # así que distribuimos los bytes solo entre los paquetes de datos
    tamanios_fwd = generar_tamanios(fwd_pkts, fwd_bytes, fwd_mean, fwd_std)
    tamanios_bwd = generar_tamanios(bwd_pkts, bwd_bytes, bwd_mean, bwd_std)
    iats_fwd     = generar_iats_direccionales(fwd_pkts, fwd_iat_mean, fwd_iat_std)
    iats_bwd     = generar_iats_direccionales(bwd_pkts, bwd_iat_mean, bwd_iat_std)

    # ── Normalizar IATs para que Flow Duration coincida exactamente ──
    # CICFlowMeter mide la duración desde el primer al último paquete,
    # que equivale a la suma de todos los IATs del flujo combinado.
    total_pkts_datos = fwd_pkts + bwd_pkts
    if flow_duration_s > 0 and total_pkts_datos > 1:
        suma_iats = sum(iats_fwd) + sum(iats_bwd)
        if suma_iats > 0:
            factor = flow_duration_s / suma_iats
            iats_fwd = [x * factor for x in iats_fwd]
            iats_bwd = [x * factor for x in iats_bwd]

    # ── Construir flags con handshake dentro del conteo ──────────────
    flags_fwd, flags_bwd = construir_listas_flags_perfectas(
        fwd_pkts, bwd_pkts, syn, fin, rst, psh, ack)

    print(f"\n[*] Reconstruyendo flujo '{nombre_ataque}' (handshake incluido)...")
    print(f"    FWD: {fwd_pkts} pkts | {fwd_bytes} bytes | win={ventana_fwd}")
    print(f"    BWD: {bwd_pkts} pkts | {bwd_bytes} bytes | win={ventana_bwd}")
    print(f"    Flow Duration CSV: {flow_duration_us:.0f} µs ({flow_duration_s:.6f} s)")
    print(f"    IAT FWD Total: {sum(iats_fwd)*1e6:.0f} µs | IAT BWD Total: {sum(iats_bwd)*1e6:.0f} µs")
    print(f"    Flags FWD: {flags_fwd}")
    print(f"    Flags BWD: {flags_bwd}")

    # ── MOTOR TCP PING-PONG ──────────────────────────────────────────
    paquetes = []
    seq_fwd = 1000
    seq_bwd = 5000
    t = time.time()
    idx_fwd = 0
    idx_bwd = 0

    bytes_pendientes_fwd = 0  # Bytes que no pudieron ir en SYN/FIN → se pasan al siguiente
    bytes_pendientes_bwd = 0

    for _ in range(fwd_pkts + bwd_pkts):
        if idx_fwd < fwd_pkts and (idx_fwd <= idx_bwd or idx_bwd >= bwd_pkts):
            flag    = flags_fwd[idx_fwd]
            es_handshake = ('S' in flag and 'A' not in flag)

            bytes_asignados = tamanios_fwd[idx_fwd]
            if es_handshake or 'F' in flag:
                bytes_pendientes_fwd += bytes_asignados  # guardamos para el siguiente
                payload = 0
            else:
                payload = bytes_asignados + bytes_pendientes_fwd
                bytes_pendientes_fwd = 0

            datos   = Raw(load=b"X" * payload) if payload > 0 else b""
            opts    = opciones_fwd if es_handshake else []
            ack_val = 0 if es_handshake else seq_bwd

            p = IP(src=IP_ATACANTE, dst=IP_VICTIMA) / \
                TCP(sport=PUERTO_ATACANTE, dport=PUERTO_VICTIMA,
                    flags=flag, seq=seq_fwd, ack=ack_val,
                    window=ventana_fwd, options=opts)
            if datos: p = p / datos
            p.time = t; paquetes.append(p)

            inc = payload + (1 if ('S' in flag or 'F' in flag) else 0)
            seq_fwd += inc  # sin max(inc,1) — TCP real no avanza seq en ACK puro
            t += iats_fwd[idx_fwd] if idx_fwd < len(iats_fwd) else 0.001
            idx_fwd += 1

        elif idx_bwd < bwd_pkts:
            flag    = flags_bwd[idx_bwd]
            es_synack = ('S' in flag and 'A' in flag)

            bytes_asignados = tamanios_bwd[idx_bwd]
            if es_synack or 'F' in flag:
                bytes_pendientes_bwd += bytes_asignados
                payload = 0
            else:
                payload = bytes_asignados + bytes_pendientes_bwd
                bytes_pendientes_bwd = 0

            datos   = Raw(load=b"V" * payload) if payload > 0 else b""
            opts    = opciones_bwd if es_synack else []

            p = IP(src=IP_VICTIMA, dst=IP_ATACANTE) / \
                TCP(sport=PUERTO_VICTIMA, dport=PUERTO_ATACANTE,
                    flags=flag, seq=seq_bwd, ack=seq_fwd,
                    window=ventana_bwd, options=opts)
            if datos: p = p / datos
            p.time = t; paquetes.append(p)

            inc = payload + (1 if ('S' in flag or 'F' in flag) else 0)
            seq_bwd += inc  # sin max(inc,1)
            t += iats_bwd[idx_bwd] if idx_bwd < len(iats_bwd) else 0.001
            idx_bwd += 1

    # ── GUARDAR ──────────────────────────────────────────────────────
    nombre_seguro  = nombre_ataque.replace(" ", "_").replace("/", "_")
    nombre_archivo = f"Ataque_Sintetizado_{nombre_seguro}_CON_HANDSHAKE.pcap"
    wrpcap(nombre_archivo, paquetes)
    print(f"\n[🏆] Guardado: '{nombre_archivo}'")
    print(f"     -> Total paquetes exactos: {len(paquetes)} (igual que el CSV)")


if __name__ == "__main__":
    fila, nombre = seleccionar_fila_aleatoria(ARCHIVO_CSV)
    generar_pcap_desde_flujo(fila, nombre)