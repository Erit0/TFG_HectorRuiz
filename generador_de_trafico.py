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
# FUNCIONES AUXILIARES (Tus algoritmos matemáticos)
# ==========================================
def inferir_opciones_tcp(header_total, n_pkts):
    if n_pkts <= 0: return []
    header_medio = header_total / n_pkts
    opciones = []
    if header_medio >= 24: opciones.append(('NOP', None))
    if header_medio >= 28: opciones.append(('MSS', 1460))
    if header_medio >= 32: opciones.append(('WScale', 7))
    if header_medio >= 36: opciones.append(('SAckOK', b''))
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

def construir_lista_flags(n_pkts, syn, fin, rst, psh, ack):
    flags = []
    restantes = {'S': int(syn), 'F': int(fin), 'R': int(rst), 'P': int(psh), 'A': int(ack)}
    for i in range(n_pkts):
        f = ''
        if restantes['S'] > 0 and i == 0:
            f += 'S'; restantes['S'] -= 1
        if restantes['F'] > 0 and i >= n_pkts - restantes['F']:
            f += 'F'; restantes['F'] -= 1
        if restantes['R'] > 0 and i == n_pkts - 1:
            f += 'R'; restantes['R'] -= 1
        if restantes['P'] > 0:
            intervalo = max(n_pkts // (restantes['P'] + 1), 1)
            if i % intervalo == 0:
                f += 'P'; restantes['P'] -= 1
        if restantes['A'] > 0 and 'S' not in f:
            f += 'A'; restantes['A'] -= 1
        if not f: f = 'A'
        flags.append(f)
    return flags

# ==========================================
# CONSTRUCCIÓN DEL PCAP (MOTOR TCP PERFECTO)
# ==========================================
def generar_pcap_desde_flujo(flujo, nombre_ataque):
    if flujo is None: return

    # ── Extraer métricas ──────────────────────────────────────────
    fwd_pkts = int(max(flujo.get('Total Fwd Packets', 1), 1))
    bwd_pkts = int(max(flujo.get('Total Backward Packets', 0), 0))
    fwd_bytes = int(max(flujo.get('Total Length of Fwd Packets', 0), 0))
    bwd_bytes = int(max(flujo.get('Total Length of Bwd Packets', 0), 0))
    ventana_fwd = int(max(flujo.get('Init_Win_bytes_forward',  8192), 1))
    ventana_bwd = int(max(flujo.get('Init_Win_bytes_backward', 8192), 1))

    fwd_mean = float(flujo.get('Fwd Packet Length Mean', fwd_bytes / fwd_pkts if fwd_pkts else 0))
    fwd_std  = float(flujo.get('Fwd Packet Length Std',  0))
    bwd_mean = float(flujo.get('Bwd Packet Length Mean', bwd_bytes / bwd_pkts if bwd_pkts else 0))
    bwd_std  = float(flujo.get('Bwd Packet Length Std',  0))

    fwd_iat_mean = float(flujo.get('Fwd IAT Mean', flujo.get('Flow IAT Mean', 1000)))
    fwd_iat_std  = float(flujo.get('Fwd IAT Std',  flujo.get('Flow IAT Std',  0)))
    bwd_iat_mean = float(flujo.get('Bwd IAT Mean', flujo.get('Flow IAT Mean', 1000)))
    bwd_iat_std  = float(flujo.get('Bwd IAT Std',  flujo.get('Flow IAT Std',  0)))

    fwd_header_total = float(flujo.get('Fwd Header Length', 20 * fwd_pkts))
    bwd_header_total = float(flujo.get('Bwd Header Length', 20 * bwd_pkts))
    opciones_fwd = inferir_opciones_tcp(fwd_header_total, fwd_pkts)
    opciones_bwd = inferir_opciones_tcp(bwd_header_total, bwd_pkts)

    # Banderas estrictamente matematicas del GAN
    syn = flujo.get('SYN Flag Count', 0)
    fin = flujo.get('FIN Flag Count', 0)
    rst = flujo.get('RST Flag Count', 0)
    psh = flujo.get('PSH Flag Count', 0)
    ack = flujo.get('ACK Flag Count', 0)

    flags_fwd = construir_lista_flags(fwd_pkts, syn, 0, rst, psh, ack)
    flags_bwd = construir_lista_flags(bwd_pkts, 0, fin, 0, 0, max(bwd_pkts, 0))

    tamanios_fwd = generar_tamanios(fwd_pkts, fwd_bytes, fwd_mean, fwd_std)
    tamanios_bwd = generar_tamanios(bwd_pkts, bwd_bytes, bwd_mean, bwd_std)
    
    iats_fwd = generar_iats_direccionales(fwd_pkts, fwd_iat_mean, fwd_iat_std)
    iats_bwd = generar_iats_direccionales(bwd_pkts, bwd_iat_mean, bwd_iat_std)

    print(f"\n[*] Reconstruyendo flujo '{nombre_ataque}' (Secuencia TCP Estricta)...")
    print(f"    FWD: {fwd_pkts} pkts | {fwd_bytes} bytes | win={ventana_fwd}")
    print(f"    BWD: {bwd_pkts} pkts | {bwd_bytes} bytes | win={ventana_bwd}")

    # ── MOTOR TCP PING-PONG ──────────────────────────────────────────
    paquetes = []
    seq_fwd = 1000 # Secuencia inicial atacante
    seq_bwd = 5000 # Secuencia inicial servidor
    t = time.time()

    idx_fwd = 0
    idx_bwd = 0
    total_pkts = fwd_pkts + bwd_pkts

    for _ in range(total_pkts):
        # Lógica Ping-Pong: Turno de FWD si no ha terminado y le toca (o BWD ya acabó)
        if idx_fwd < fwd_pkts and (idx_fwd <= idx_bwd or idx_bwd >= bwd_pkts):
            flag = flags_fwd[idx_fwd]
            payload = tamanios_fwd[idx_fwd]
            datos = Raw(load=b"X" * payload) if payload > 0 else b""

            p = IP(src=IP_ATACANTE, dst=IP_VICTIMA) / \
                TCP(sport=PUERTO_ATACANTE, dport=PUERTO_VICTIMA,
                    flags=flag, seq=seq_fwd, ack=seq_bwd,
                    window=ventana_fwd, options=opciones_fwd)
            
            if datos: p = p / datos
            p.time = t
            paquetes.append(p)

            # MATEMÁTICA TCP REAL: Solo avanzamos la secuencia según la carga real
            seq_increment = payload
            if 'S' in flag or 'F' in flag: 
                seq_increment += 1 # SYN y FIN consumen 1 byte de secuencia virtual
            
            seq_fwd += seq_increment
            t += iats_fwd[idx_fwd] if idx_fwd < len(iats_fwd) else 0.001
            idx_fwd += 1

        # Turno de BWD
        elif idx_bwd < bwd_pkts:
            flag = flags_bwd[idx_bwd]
            payload = tamanios_bwd[idx_bwd]
            datos = Raw(load=b"V" * payload) if payload > 0 else b""

            # Si BWD responde, nos aseguramos de que lleve la bandera ACK encendida
            if 'A' not in flag and 'S' not in flag: flag += 'A'

            p = IP(src=IP_VICTIMA, dst=IP_ATACANTE) / \
                TCP(sport=PUERTO_VICTIMA, dport=PUERTO_ATACANTE,
                    flags=flag, seq=seq_bwd, ack=seq_fwd, # Fíjate cómo ack escucha al seq del atacante
                    window=ventana_bwd, options=opciones_bwd)
            
            if datos: p = p / datos
            p.time = t
            paquetes.append(p)

            seq_increment = payload
            if 'S' in flag or 'F' in flag:
                seq_increment += 1
                
            seq_bwd += seq_increment
            t += iats_bwd[idx_bwd] if idx_bwd < len(iats_bwd) else 0.001
            idx_bwd += 1

    # ── GUARDAR ──────────────────────────────────────────────────────────
    nombre_seguro = nombre_ataque.replace(" ", "_").replace("/", "_")
    nombre_archivo = f"Ataque_Sintetizado_{nombre_seguro}_PERFECTO.pcap"
    wrpcap(nombre_archivo, paquetes)
    print(f"\n[🏆] Guardado: '{nombre_archivo}'")
    print(f"     -> Paquetes generados exactamente como indica el CSV: {len(paquetes)}")

if __name__ == "__main__":
    fila, nombre = seleccionar_fila_aleatoria(ARCHIVO_CSV)
    generar_pcap_desde_flujo(fila, nombre)