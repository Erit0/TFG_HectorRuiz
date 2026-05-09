"""
generador_de_trafico.py
===============
Genera una PCAP realista por flujo TCP a partir del CSV adversario.

Lógica de temporización (PING-PONG real):
  - Los paquetes Fwd y Bwd se entrelazan en orden cronológico real.
  - Cada dirección lleva su propio cursor de tiempo que avanza según
    sus IATs direccionales (Fwd IAT Mean/Std y Bwd IAT Mean/Std).
  - t_inicio_bwd se calcula para que el último paquete Bwd caiga
    exactamente en t0 + Flow_Duration del CSV.
  - El handshake (SYN/SYN-ACK/ACK) solo se genera si SYN Flag Count > 0
    en el CSV, respetando flujos capturados a mitad de sesión.

Fidelidad a act_data_pkt_fwd:
  - Los bytes del flujo se distribuyen solo entre los paquetes que
    CICFlowMeter marca como portadores de payload (act_data_pkt_fwd).
  - El resto de paquetes fwd se generan como ACKs puros (0 bytes),
    produciendo los Duplicate ACKs que corresponden al tráfico real.
  - Si act_data_pkt_fwd no está en el CSV, se usa fwd_pkts como fallback.

Uso:
    python generador_de_trafico.py                        # fila aleatoria
    python generador_de_trafico.py --idx 5                # fila concreta
    python generador_de_trafico.py --label DDoS           # tipo de ataque aleatorio
    python generador_de_trafico.py --all --max 20         # hasta 20 flujos, uno por fila
"""

import argparse
import os
import time
import random

import numpy as np
import pandas as pd

import scapy.config
scapy.config.conf.ipv6_enabled = False
from scapy.layers.inet import IP, TCP
from scapy.packet import Raw
from scapy.utils import wrpcap

# ──────────────────────────────────────────────
# CONFIGURACIÓN DE RED
# ──────────────────────────────────────────────
IP_ATACANTE    = "192.168.1.50"
IP_VICTIMA     = "10.0.0.200"
PUERTO_VICTIMA = 80
MSS            = 1460  # Maximum Segment Size estándar TCP

ARCHIVO_CSV = "ataques_adversarios_COMPLETOS.csv"

PUERTOS_POR_LABEL = {
    "ftp-patator": 21,
    "ssh-patator":  22,
    "portscan":    443,
    "bot":         443,
}

# ──────────────────────────────────────────────
# COMPORTAMIENTO TCP POR LABEL (independiente del CSV)
# Cada entrada define el patrón de apertura/cierre real del ataque.
#
#   syn  : True  → siempre generar SYN/SYN-ACK/ACK al inicio
#   fin  : True  → cerrar con FIN/ACK (conexión completa)
#   rst  : True  → cerrar con RST     (conexión abortada)
#
# Prioridad de cierre: rst > fin > sin cierre (conexión colgada)
# ──────────────────────────────────────────────
TCP_BEHAVIOR = {
    # SYN + FIN: conexión TCP completa
    "benign":                    {"syn": True,  "fin": True,  "rst": False},
    "ftp-patator":               {"syn": True,  "fin": True,  "rst": False},
    "ssh-patator":               {"syn": True,  "fin": True,  "rst": False},
    "web attack brute force":    {"syn": True,  "fin": True,  "rst": False},
    "web attack xss":            {"syn": True,  "fin": True,  "rst": False},
    "web attack sql injection":  {"syn": True,  "fin": True,  "rst": False},
    "heartbleed":                {"syn": True,  "fin": True,  "rst": False},
    "infiltration":              {"syn": True,  "fin": True,  "rst": False},
    "bot":                       {"syn": True,  "fin": True,  "rst": False},
    # SYN + RST: conexión abortada
    "dos hulk":                  {"syn": True,  "fin": False, "rst": True},
    "dos goldeneye":             {"syn": True,  "fin": False, "rst": True},
    "portscan":                  {"syn": True,  "fin": False, "rst": True},
    # SYN solo: conexión intencionalmente incompleta (se queda colgada)
    "ddos":                      {"syn": True,  "fin": False, "rst": False},
    "dos slowloris":             {"syn": True,  "fin": False, "rst": False},
    "dos slowhttptest":          {"syn": True,  "fin": False, "rst": False},
}

def tcp_behavior_para_label(label: str) -> dict:
    """
    Devuelve el comportamiento TCP correcto para el label dado.
    Busca por subcadena (case-insensitive) para tolerar variantes
    del CSV como 'Web Attack  Brute Force' o 'DoS Hulk'.
    Fallback: SYN + FIN (conexión completa) si el label no está mapeado.
    """
    label_lower = label.lower().strip()
    for key, behavior in TCP_BEHAVIOR.items():
        if key in label_lower:
            return behavior
    # Label desconocido → conexión completa por defecto
    return {"syn": True, "fin": True, "rst": False}


# ──────────────────────────────────────────────
# FILTRO DE APLICABILIDAD DEL CSV
# ──────────────────────────────────────────────
def filtrar_flujos_generables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Descarta filas que producirían PCAPs incoherentes o inválidas.

    Criterios de descarte:
      1. Init_Win_bytes_forward <= 0  → window TCP inválida en el SYN
      2. Flow Packets/s >= 1e9        → artefacto de CICFlowMeter (div/0)
      3. Flow Duration == 0 con >1 paquete → flujo instantáneo imposible
      4. Flow IAT Min < 0             → timestamp negativo, bug del extractor
      5. Payload Fwd == 0 sin SYN     → sin contenido ni handshake
    """
    n_antes = len(df)

    mask_ok = (
        (df["Init_Win_bytes_forward"] > 0)
        & (df["Flow Packets/s"] < 1e9)
        & ~((df["Flow Duration"] == 0) & (df["Total Fwd Packets"] > 1))
        & (df["Flow IAT Min"] >= 0)
        & (df["Total Fwd Packets"] > 0)
    )

    df_ok = df[mask_ok].copy().reset_index(drop=True)
    descartados = n_antes - len(df_ok)
    if descartados:
        print(f"[filtro] {n_antes} → {len(df_ok)} flujos "
              f"({descartados} descartados, {descartados/n_antes*100:.1f}%)")
    return df_ok


# ──────────────────────────────────────────────
# DISTRIBUCIONES DE IAT
# ──────────────────────────────────────────────
def _elegir_distribucion(mean_us: float, std_us: float, n: int) -> np.ndarray:
    """
    Selecciona la distribución según el coeficiente de variación (CV = std/mean).

    CV < 0.3 → Weibull k>1   (tráfico periódico: keepalives, heartbeats)
    CV ≈ 1   → Exponencial   (Poisson-like: DoS floods, port scans)
    CV > 1   → Log-normal    (cola larga: HTTP idle, SSH interactivo)
    """
    if std_us <= 0 or n <= 1:
        return np.full(n, mean_us)

    cv = std_us / mean_us

    if cv < 0.3:
        import math
        k   = max(1.0 / (cv ** 1.086), 1.5)
        lam = mean_us / math.gamma(1.0 + 1.0 / k)
        raw = np.random.weibull(k, n) * lam
    elif cv < 1.3:
        raw = np.random.exponential(scale=mean_us, size=n)
    else:
        sigma2 = np.log(1.0 + cv ** 2)
        mu     = np.log(mean_us) - sigma2 / 2.0
        raw    = np.random.lognormal(mean=mu, sigma=np.sqrt(sigma2), size=n)

    return raw


def generar_iats(n_pkts: int, mean_us: float, std_us: float,
                 total_us: float) -> list:
    """
    Genera n_pkts-1 intervalos (segundos) con la distribución correcta,
    escalados para que su suma == total_us µs exacto.
    """
    n_intervalos = max(n_pkts - 1, 0)
    if n_intervalos == 0 or mean_us <= 0:
        return []

    raw  = _elegir_distribucion(mean_us, std_us, n_intervalos)
    raw  = np.clip(raw, 1.0, None)
    suma = raw.sum()
    if suma > 0:
        raw = raw * (total_us / suma)

    return (raw / 1_000_000.0).tolist()


# ──────────────────────────────────────────────
# DISTRIBUCIÓN DE TAMAÑOS DE PAYLOAD
# ──────────────────────────────────────────────
def distribuir_bytes(n_pkts: int, total_bytes: int,
                     mean: float, std: float) -> list:
    """
    Reparte total_bytes entre n_pkts paquetes siguiendo una distribución
    normal truncada en 0. El último paquete absorbe el redondeo.
    """
    if n_pkts <= 0:
        return []
    if total_bytes <= 0:
        return [0] * n_pkts
    if n_pkts == 1:
        return [total_bytes]

    if std > 0:
        raw = np.random.normal(mean, min(std, mean * 0.8), n_pkts)
        raw = np.clip(raw, 0.0, None)
        s   = raw.sum()
        raw = raw * (total_bytes / s) if s > 0 else np.full(n_pkts, total_bytes / n_pkts)
    else:
        raw = np.full(n_pkts, total_bytes / n_pkts)

    sizes = np.round(raw).astype(int)
    sizes[-1] = max(0, sizes[-1] + (total_bytes - sizes.sum()))
    return sizes.tolist()


def fragmentar_si_necesario(sizes: list) -> list:
    """
    Fragmenta cualquier segmento que supere MSS en trozos de MSS,
    manteniendo el total de bytes idéntico.
    Esto evita paquetes IP imposibles en ataques con pocos act_data
    pero muchos bytes (ej. DoS Hulk: 1 paquete de datos con 10 KB).
    """
    resultado = []
    for sz in sizes:
        while sz > MSS:
            resultado.append(MSS)
            sz -= MSS
        resultado.append(sz)
    return resultado


# ──────────────────────────────────────────────
# OPCIONES TCP REALISTAS
# ──────────────────────────────────────────────
def opciones_syn(mss: int = 1460, wscale: int = 7, ts_val: int = None) -> list:
    if ts_val is None:
        ts_val = int(time.time() * 1000) & 0xFFFFFFFF
    return [
        ("MSS",       mss),
        ("SAckOK",    b""),
        ("Timestamp", (ts_val, 0)),
        ("NOP",       None),
        ("WScale",    wscale),
    ]


def opciones_synack(mss: int = 1460, wscale: int = 7,
                    ts_val: int = None, ts_echo: int = 0) -> list:
    if ts_val is None:
        ts_val = int(time.time() * 1000) & 0xFFFFFFFF
    return [
        ("MSS",       mss),
        ("SAckOK",    b""),
        ("Timestamp", (ts_val, ts_echo)),
        ("NOP",       None),
        ("WScale",    wscale),
    ]


def opciones_datos(ts_val: int, ts_echo: int) -> list:
    return [
        ("NOP",       None),
        ("NOP",       None),
        ("Timestamp", (ts_val & 0xFFFFFFFF, ts_echo & 0xFFFFFFFF)),
    ]


# ──────────────────────────────────────────────
# ASIGNACIÓN DE FLAGS TCP
# ──────────────────────────────────────────────
def asignar_flags(fwd_pkts: int, bwd_pkts: int,
                  fin_cnt: int, rst_cnt: int, psh_cnt: int,
                  syn_cnt: int = 0):
    """
    Construye las listas de flags por paquete para Fwd y Bwd.

    El comportamiento depende de la combinación syn/fin/rst forzada por label:

    SYN + FIN (conexión completa):
        SYN → SYN-ACK → ACK ... datos ... FIN-ACK → FIN-ACK
    SYN + RST (conexión abortada):
        SYN → SYN-ACK → ACK ... datos ... RST
        (sin FIN en ninguna dirección)
    SYN solo (conexión intencionalmente incompleta):
        SYN → SYN-ACK → ACK ... datos ...
        (sin cierre: slowloris, DDoS SYN flood)
    Sin SYN (mid-session):
        Solo ACKs y datos, sin handshake ni cierre explícito.

    PSH se distribuye uniformemente en la zona de datos Fwd.
    """
    flags_fwd = ["A"] * fwd_pkts
    flags_bwd = ["A"] * bwd_pkts

    hw_fwd, hw_bwd = 0, 0   # offset tras handshake
    cierre_fwd = 0           # offset al cierre al final

    if syn_cnt > 0:
        # ── Handshake: SYN → SYN-ACK → ACK ──────────────────────────────
        if fwd_pkts >= 2 and bwd_pkts >= 1:
            flags_fwd[0] = "S"
            flags_bwd[0] = "SA"
            flags_fwd[1] = "A"
            hw_fwd, hw_bwd = 2, 1
        elif fwd_pkts >= 1:
            flags_fwd[0] = "S"
            hw_fwd = 1

        # ── Cierre según comportamiento del ataque ────────────────────────
        if rst_cnt > 0 and fin_cnt == 0:
            # SYN + RST: solo RST al final del lado fwd, sin FIN
            if fwd_pkts > hw_fwd:
                flags_fwd[-1] = "R"
                cierre_fwd = 1
            # bwd no cierra con nada (RST aborta sin negociación)

        elif fin_cnt > 0:
            # SYN + FIN: cierre limpio en ambas direcciones
            if fwd_pkts > hw_fwd:
                flags_fwd[-1] = "FA"
                cierre_fwd = 1
            if bwd_pkts > hw_bwd:
                flags_bwd[-1] = "FA"

        # else: SYN solo → sin cierre (slowloris, DDoS); cierre_fwd = 0

    # ── PSH distribuido en zona de datos Fwd ─────────────────────────────
    # Excluye handshake y el paquete de cierre (FIN/RST)
    zona      = list(range(hw_fwd, fwd_pkts - cierre_fwd))
    restantes = int(psh_cnt)
    if zona and restantes > 0:
        paso = max(len(zona) // (restantes + 1), 1)
        for i, pos in enumerate(zona):
            if restantes <= 0:
                break
            if i % paso == 0:
                flags_fwd[pos] = "PA"
                restantes -= 1

    return flags_fwd, flags_bwd


# ──────────────────────────────────────────────
# CONSTRUCCIÓN DEL FLUJO DE PAQUETES
# ──────────────────────────────────────────────
def construir_paquetes(flujo: pd.Series, t0: float) -> list:
    label = str(flujo.get("Label", "unknown")).lower()

    sport = random.randint(49152, 65535)
    dport = PUERTO_VICTIMA
    for k, v in PUERTOS_POR_LABEL.items():
        if k in label:
            dport = v
            break

    fwd_pkts  = int(max(flujo.get("Total Fwd Packets",      1), 1))
    bwd_pkts  = int(max(flujo.get("Total Backward Packets", 0), 0))
    fwd_bytes = int(max(flujo.get("Total Length of Fwd Packets", 0), 0))
    bwd_bytes = int(max(flujo.get("Total Length of Bwd Packets", 0), 0))

    fwd_mean = float(flujo.get("Fwd Packet Length Mean",
                                fwd_bytes / fwd_pkts if fwd_pkts else 0))
    fwd_std  = float(flujo.get("Fwd Packet Length Std", 0))
    bwd_mean = float(flujo.get("Bwd Packet Length Mean",
                                bwd_bytes / bwd_pkts if bwd_pkts else 0))
    bwd_std  = float(flujo.get("Bwd Packet Length Std", 0))

    win_fwd = int(np.clip(flujo.get("Init_Win_bytes_forward",  8192), 1, 65535))
    win_bwd = int(np.clip(flujo.get("Init_Win_bytes_backward", 8192), 1, 65535))

    fwd_iat_total = float(flujo.get("Fwd IAT Total", 0))
    fwd_iat_mean  = float(flujo.get("Fwd IAT Mean",  0))
    fwd_iat_std   = float(flujo.get("Fwd IAT Std",   0))
    bwd_iat_total = float(flujo.get("Bwd IAT Total", 0))
    bwd_iat_mean  = float(flujo.get("Bwd IAT Mean",  0))
    bwd_iat_std   = float(flujo.get("Bwd IAT Std",   0))

    psh     = int(flujo.get("PSH Flag Count", 0))

    # ── Comportamiento TCP forzado por label (ignora valores del CSV) ─────
    # Los flags SYN/FIN/RST se determinan por el tipo de ataque real,
    # no por lo que reportó CICFlowMeter en el CSV original.
    behavior = tcp_behavior_para_label(label)
    syn_cnt  = 1 if behavior["syn"] else 0
    fin      = 1 if behavior["fin"] else 0
    rst      = 1 if behavior["rst"] else 0

    # ── Distribución de payload fiel a act_data_pkt_fwd ──────────────────
    #
    # act_data_pkt_fwd (feature de CICFlowMeter) indica cuántos paquetes
    # fwd transportaron payload real. El resto son ACKs puros (0 bytes).
    # Respetar este valor hace que los Duplicate ACKs del PCAP coincidan
    # con los que habría en una captura real del mismo tráfico.
    #
    # Fallback: si la columna no existe en el CSV, se usan todos los pkts.
    act_data = int(flujo.get("act_data_pkt_fwd", fwd_pkts))
    act_data = max(1, min(act_data, fwd_pkts))  # sanidad: [1, fwd_pkts]

    if act_data < fwd_pkts:
        # Distribuir bytes solo entre act_data paquetes y fragmentar si >MSS
        sizes_data = distribuir_bytes(act_data, fwd_bytes, fwd_mean, fwd_std)
        sizes_data = fragmentar_si_necesario(sizes_data)

        # Si la fragmentación añadió paquetes extra, ajustar fwd_pkts
        pkts_datos_real = len(sizes_data)
        pkts_ack_puros  = fwd_pkts - act_data  # ACKs sin payload

        if pkts_datos_real + pkts_ack_puros != fwd_pkts:
            # La fragmentación generó más paquetes de los previstos;
            # recalcular el total para mantener coherencia
            fwd_pkts = pkts_datos_real + pkts_ack_puros

        sizes_fwd = sizes_data + [0] * pkts_ack_puros
        # Mezclar: los ACKs no van todos al final, sino intercalados
        random.shuffle(sizes_fwd)
    else:
        # Todos los paquetes llevan datos (act_data == fwd_pkts)
        sizes_fwd = distribuir_bytes(fwd_pkts, fwd_bytes, fwd_mean, fwd_std)
        sizes_fwd = fragmentar_si_necesario(sizes_fwd)
        if len(sizes_fwd) != fwd_pkts:
            fwd_pkts = len(sizes_fwd)

    sizes_bwd = distribuir_bytes(bwd_pkts, bwd_bytes, bwd_mean, bwd_std)
    sizes_bwd = fragmentar_si_necesario(sizes_bwd)
    if len(sizes_bwd) != bwd_pkts:
        bwd_pkts = len(sizes_bwd)

    # ── IATs ──────────────────────────────────────────────────────────────
    iats_fwd = generar_iats(fwd_pkts, fwd_iat_mean, fwd_iat_std, fwd_iat_total)
    iats_bwd = generar_iats(bwd_pkts, bwd_iat_mean, bwd_iat_std, bwd_iat_total)

    # ── Flags TCP ─────────────────────────────────────────────────────────
    flags_fwd, flags_bwd = asignar_flags(
        fwd_pkts, bwd_pkts, fin, rst, psh, syn_cnt=syn_cnt
    )

    # ── Timestamps ────────────────────────────────────────────────────────
    # t_inicio_bwd se ancla algebraicamente para que el último paquete
    # Bwd caiga exactamente en t0 + Flow_Duration del CSV.
    #
    #   ts_bwd[-1] = t0 + t_inicio_bwd + bwd_iat_total = t0 + flow_duration_s
    #   → t_inicio_bwd = flow_duration_s - bwd_iat_total
    #
    # Fallback para flujos unidireccionales (bwd_pkts == 0 o bwd_total == 0).
    flow_duration_s = float(flujo.get("Flow Duration", 0)) / 1_000_000.0
    bwd_total_s     = bwd_iat_total / 1_000_000.0

    if bwd_pkts > 0 and bwd_total_s > 0:
        t_inicio_bwd = max(flow_duration_s - bwd_total_s, 0.0)
    else:
        t_inicio_bwd = (iats_fwd[0] / 2.0) if iats_fwd else 0.0005

    ts_cursor_fwd = [t0]
    for iat in iats_fwd:
        ts_cursor_fwd.append(ts_cursor_fwd[-1] + iat)

    ts_cursor_bwd = [t0 + t_inicio_bwd]
    for iat in iats_bwd:
        ts_cursor_bwd.append(ts_cursor_bwd[-1] + iat)

    # ── Opciones TCP ──────────────────────────────────────────────────────
    ts_base       = int(t0 * 1000) & 0xFFFFFFFF
    ts_syn_opt    = ts_base
    ts_synack_opt = (ts_base + 1) & 0xFFFFFFFF

    opts_syn    = opciones_syn(ts_val=ts_syn_opt)
    opts_synack = opciones_synack(ts_val=ts_synack_opt, ts_echo=ts_syn_opt)

    # ── Números de secuencia ──────────────────────────────────────────────
    isn_fwd = random.randint(100_000, 999_999_999)
    isn_bwd = random.randint(100_000, 999_999_999)
    seq_fwd = isn_fwd
    seq_bwd = isn_bwd

    # ── Merge ping-pong por timestamp ─────────────────────────────────────
    eventos = []
    for i, ts in enumerate(ts_cursor_fwd[:fwd_pkts]):
        eventos.append((ts, "fwd", i))
    for i, ts in enumerate(ts_cursor_bwd[:bwd_pkts]):
        eventos.append((ts, "bwd", i))
    eventos.sort(key=lambda e: e[0])

    paquetes = []

    for ts, direccion, idx in eventos:
        ts_opt_val = (int(ts * 1000)) & 0xFFFFFFFF

        if direccion == "fwd":
            flag       = flags_fwd[idx]
            es_syn     = (flag == "S")
            payload_sz = sizes_fwd[idx] if not es_syn else 0
            opts       = opts_syn if es_syn else opciones_datos(ts_opt_val, ts_synack_opt)

            pkt = (IP(src=IP_ATACANTE, dst=IP_VICTIMA) /
                   TCP(sport=sport, dport=dport,
                       flags=flag,
                       seq=seq_fwd,
                       ack=(0 if es_syn else seq_bwd),
                       window=win_fwd,
                       options=opts))
            if payload_sz > 0:
                pkt = pkt / Raw(load=b"\x00" * payload_sz)

            pkt.time = ts
            paquetes.append(pkt)

            inc     = payload_sz + (1 if ("S" in flag or "F" in flag) else 0)
            seq_fwd = (seq_fwd + inc) & 0xFFFFFFFF

        else:
            flag      = flags_bwd[idx]
            es_synack = ("S" in flag and "A" in flag)
            payload_sz = sizes_bwd[idx] if not es_synack else 0
            opts       = opts_synack if es_synack else opciones_datos(ts_opt_val, ts_syn_opt)

            pkt = (IP(src=IP_VICTIMA, dst=IP_ATACANTE) /
                   TCP(sport=dport, dport=sport,
                       flags=flag,
                       seq=seq_bwd,
                       ack=seq_fwd,
                       window=win_bwd,
                       options=opts))
            if payload_sz > 0:
                pkt = pkt / Raw(load=b"\x00" * payload_sz)

            pkt.time = ts
            paquetes.append(pkt)

            inc     = payload_sz + (1 if ("S" in flag or "F" in flag) else 0)
            seq_bwd = (seq_bwd + inc) & 0xFFFFFFFF

    return paquetes


# ──────────────────────────────────────────────
# CARGA Y SELECCIÓN DE FILAS
# ──────────────────────────────────────────────
def cargar_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No encontrado: {path}")
    df = pd.read_csv(path)
    if "Label" in df.columns:
        df["Label"] = (df["Label"].astype(str)
                       .str.encode("ascii", "ignore")
                       .str.decode("ascii")
                       .str.strip())
    df = filtrar_flujos_generables(df)
    return df


def seleccionar_fila(df: pd.DataFrame, idx: int = None,
                     label: str = None) -> pd.Series:
    if idx is not None:
        return df.iloc[idx]
    if label:
        sub = df[df["Label"].str.lower() == label.lower()]
        if sub.empty:
            raise ValueError(f"No hay filas con label='{label}'")
        return sub.sample(1).iloc[0]
    return df.sample(1).iloc[0]


# ──────────────────────────────────────────────
# INFORME POR CONSOLA
# ──────────────────────────────────────────────
def imprimir_resumen(flujo: pd.Series, n_pkts: int,
                     nombre_archivo: str) -> None:
    fwd      = int(flujo.get("Total Fwd Packets",      0))
    bwd      = int(flujo.get("Total Backward Packets", 0))
    dur      = float(flujo.get("Flow Duration",        0))
    act_data = int(flujo.get("act_data_pkt_fwd",       fwd))
    label    = str(flujo.get("Label", "unknown"))

    behavior = tcp_behavior_para_label(label)
    if behavior["fin"]:
        tcp_modo = "SYN + FIN  (conexión completa)"
    elif behavior["rst"]:
        tcp_modo = "SYN + RST  (conexión abortada)"
    elif behavior["syn"]:
        tcp_modo = "SYN solo   (conexión incompleta)"
    else:
        tcp_modo = "Mid-session (sin handshake)"

    print("\n" + "=" * 55)
    print(f"  Ataque      : {label}")
    print(f"  Origen      : {flujo.get('_ORIGIN_FILE_', '?')}")
    print(f"  TCP modo    : {tcp_modo}")
    print(f"  Fwd pkts    : {fwd}  (con datos: {act_data}, ACK puros: {fwd - act_data})")
    print(f"  Bwd pkts    : {bwd}")
    print(f"  Fwd bytes   : {int(flujo.get('Total Length of Fwd Packets', 0))}"
          f"   |   Bwd bytes: {int(flujo.get('Total Length of Bwd Packets', 0))}")
    print(f"  Flow Dur    : {dur/1e6:.6f} s  ({dur:.0f} µs)")
    print(f"  Flow IAT Mean CSV : {float(flujo.get('Flow IAT Mean', 0)):.1f} µs")
    print(f"  Fwd IAT Mean CSV  : {float(flujo.get('Fwd IAT Mean',  0)):.1f} µs")
    print(f"  Bwd IAT Mean CSV  : {float(flujo.get('Bwd IAT Mean',  0)):.1f} µs")
    print(f"  Win fwd     : {int(flujo.get('Init_Win_bytes_forward',  0))}"
          f"   |   Win bwd: {int(flujo.get('Init_Win_bytes_backward', 0))}")
    print(f"  Paquetes generados: {n_pkts}")
    print(f"  Guardado en : {nombre_archivo}")
    print("=" * 55 + "\n")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def procesar_flujo(flujo: pd.Series, t0: float,
                   directorio: str = ".") -> str:
    label_safe = (str(flujo.get("Label", "unknown"))
                  .replace(" ", "_").replace("/", "_")
                  .encode("ascii", "ignore").decode())
    origin_idx = str(flujo.get("_ORIGIN_IDX_", "noIdx")).replace(" ", "")
    nombre = f"flujo_{label_safe}_{origin_idx}_{int(t0)}.pcap"
    ruta   = os.path.join(directorio, nombre)

    pkts = construir_paquetes(flujo, t0)
    wrpcap(ruta, pkts)
    imprimir_resumen(flujo, len(pkts), ruta)
    return ruta


def main():
    parser = argparse.ArgumentParser(
        description="Genera PCAP realista desde el CSV adversario.")
    parser.add_argument("--csv",   default=ARCHIVO_CSV)
    parser.add_argument("--idx",   type=int,  default=None,
                        help="Índice de fila concreto")
    parser.add_argument("--label", type=str,  default=None,
                        help="Tipo de ataque (ej: DDoS, PortScan)")
    parser.add_argument("--all",   action="store_true",
                        help="Generar un PCAP por cada fila del CSV")
    parser.add_argument("--max",   type=int,  default=10,
                        help="Máximo de flujos con --all (default: 10)")
    parser.add_argument("--out",   default=".",
                        help="Directorio de salida (default: .)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = cargar_csv(args.csv)
    t0 = time.time()

    if args.all:
        filas = df.head(args.max)
        print(f"[*] Generando {len(filas)} flujos...")
        for i, (_, fila) in enumerate(filas.iterrows()):
            procesar_flujo(fila, t0 + i * 200.0, args.out)
    else:
        fila = seleccionar_fila(df, idx=args.idx, label=args.label)
        procesar_flujo(fila, t0, args.out)


if __name__ == "__main__":
    main()